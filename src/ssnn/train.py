from __future__ import annotations
import time
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from .accounting import count_params, count_active_params, estimate_train_flops, estimate_forward_flops
from .tasks import batchify, Family

def evaluate_detailed(
    model: torch.nn.Module,
    rows: List[Tuple[Tuple[int, ...], int, int]],
    family: Family,
    device: torch.device,
    batch_size: int = 512,
) -> Dict[str, float]:
    """Evaluate both per-example accuracy and whole-program exact-match accuracy.

    All metrics are computed strictly from model forward outputs.
    """
    model.eval()
    hits = 0
    total = 0
    prog_correct = defaultdict(int)
    prog_total = defaultdict(int)

    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            b = rows[start:start + batch_size]
            prog, x, y, lengths = batchify(b, family.n_ops, family.n_states)
            prog = prog.to(device)
            x = x.to(device)
            y = y.to(device)
            lengths = lengths.to(device)

            logits = model(prog, x, lengths)
            preds = logits.argmax(-1)
            correct_mask = (preds == y).cpu().numpy()

            hits += int(correct_mask.sum())
            total += len(y)

            for item, is_correct in zip(b, correct_mask):
                p = item[0]
                prog_total[p] += 1
                if is_correct:
                    prog_correct[p] += 1

    example_acc = hits / max(total, 1)
    solved_progs = sum(1 for p in prog_total if prog_correct[p] == prog_total[p])
    prog_acc = solved_progs / max(len(prog_total), 1)

    return {
        "example_accuracy": example_acc,
        "program_exact_accuracy": prog_acc,
        "total_examples": total,
        "total_programs": len(prog_total),
    }


def evaluate(model: torch.nn.Module, rows, family: Family, device: torch.device, batch_size: int = 512) -> float:
    """Backward-compatible evaluation function returning per-example accuracy."""
    res = evaluate_detailed(model, rows, family, device, batch_size)
    return res["example_accuracy"]


def train(
    model: torch.nn.Module,
    train_rows: List[Tuple[Tuple[int, ...], int, int]],
    family: Family,
    device: torch.device,
    steps: int = 500,
    batch_size: int = 128,
    lr: float = 3e-3,
    seed: int = 0,
    kind: str | None = None,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    model_kind = kind or ("mlp" if model.__class__.__name__ == "MonolithicMLP" else getattr(model, "mode", "unknown"))

    start = time.time()
    losses: List[float] = []

    for step in range(steps):
        idx = rng.integers(0, len(train_rows), size=batch_size)
        rows = [train_rows[i] for i in idx]
        prog, x, y, lengths = batchify(rows, family.n_ops, family.n_states)
        prog = prog.to(device)
        x = x.to(device)
        y = y.to(device)
        lengths = lengths.to(device)

        opt.zero_grad(set_to_none=True)
        logits = model(prog, x, lengths)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()

        if step % max(1, steps // 10) == 0:
            losses.append(float(loss.detach().cpu()))

    elapsed = time.time() - start

    return {
        "seconds": elapsed,
        "loss_samples": losses,
        "params": count_params(model),
        "active_params": count_active_params(model, model_kind),
        "estimated_train_flops": estimate_train_flops(model, batch_size, steps, task_depth=3, kind=model_kind),
        "estimated_step_fwd_flops": estimate_forward_flops(model, batch_size, task_depth=3, kind=model_kind),
    }


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
