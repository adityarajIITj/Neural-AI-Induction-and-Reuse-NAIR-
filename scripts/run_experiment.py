#!/usr/bin/env python3
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from ssnn.tasks import make_family, make_split, sample_examples
from ssnn.models import build_model
from ssnn.train import train, evaluate_detailed, set_seed

def main():
    ap = argparse.ArgumentParser(
        description="NAIR Research Harness - Compositional Experiment Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--family", default="affine", choices=["affine", "permutation"], help="Deterministic task family to evaluate")
    ap.add_argument("--steps", type=int, default=500, help="Number of training optimization steps")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0], help="List of random seeds for multi-run replication")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Computation device (auto detects CUDA)")
    ap.add_argument("--models", nargs="+", default=["mlp", "shared", "explicit", "library", "soft_library", "hypernet"], help="Subset of model architectures to train")
    ap.add_argument("--hidden", type=int, default=64, help="Hidden dimension size for linear layers")
    ap.add_argument("--batch-size", type=int, default=128, help="Batch size for stochastic gradient steps")
    ap.add_argument("--train-per-program", type=int, default=24, help="Number of sampled input states per training program")
    ap.add_argument("--tag", default="pilot", help="Identifier prefix for saving run directories")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")

    family = make_family(args.family)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.tag}_{args.family}_{run_timestamp}"
    run_dir = ROOT / "results" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_id": run_id,
        "family": args.family,
        "n_ops": family.n_ops,
        "n_states": family.n_states,
        "steps": args.steps,
        "seeds": args.seeds,
        "device": str(device),
        "models": args.models,
        "hidden": args.hidden,
        "batch_size": args.batch_size,
        "train_per_program": args.train_per_program,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    all_results = []

    for seed in args.seeds:
        set_seed(seed)
        split = make_split(family, seed=seed)

        # Generate sample rows
        train_rows = sample_examples(family, split.train_programs, per_program=args.train_per_program, seed=seed)
        val_seen_rows = sample_examples(family, split.val_seen_programs, per_program=family.n_states, seed=seed + 100, exhaustive=True)
        val_heldout_rows = sample_examples(family, split.val_heldout_programs, per_program=family.n_states, seed=seed + 200, exhaustive=True)
        seen_d3_rows = sample_examples(family, split.seen_test_programs[:100], per_program=family.n_states, seed=seed + 1000, exhaustive=True)
        unseen_pair_d3_rows = sample_examples(family, split.unseen_test_programs[:100], per_program=family.n_states, seed=seed + 2000, exhaustive=True)
        strict_functional_rows = sample_examples(family, split.strict_functional_holdout_programs[:100], per_program=family.n_states, seed=seed + 3000, exhaustive=True)
        leaked_functional_rows = sample_examples(family, split.functional_leaked_programs[:100], per_program=family.n_states, seed=seed + 4000, exhaustive=True)
        depth4_rows = sample_examples(family, split.depth4_extrapolation_programs[:100], per_program=family.n_states, seed=seed + 5000, exhaustive=True)

        for kind in args.models:
            set_seed(seed)
            model = build_model(kind, family.n_ops, family.n_states, hidden=args.hidden, max_depth=4)
            fit = train(model, train_rows, family, device, steps=args.steps, batch_size=args.batch_size, seed=seed, kind=kind)

            # Evaluate on all protocol slices
            eval_val_seen = evaluate_detailed(model, val_seen_rows, family, device)
            eval_val_heldout = evaluate_detailed(model, val_heldout_rows, family, device)
            eval_seen_d3 = evaluate_detailed(model, seen_d3_rows, family, device)
            eval_unseen_d3 = evaluate_detailed(model, unseen_pair_d3_rows, family, device)
            eval_strict = evaluate_detailed(model, strict_functional_rows, family, device)
            eval_leaked = evaluate_detailed(model, leaked_functional_rows, family, device)
            eval_d4 = evaluate_detailed(model, depth4_rows, family, device)

            seen_acc = eval_seen_d3["example_accuracy"]
            unseen_acc = eval_unseen_d3["example_accuracy"]
            strict_acc = eval_strict["example_accuracy"]
            leaked_acc = eval_leaked["example_accuracy"]
            d4_acc = eval_d4["example_accuracy"]

            res = {
                "run_id": run_id,
                "seed": seed,
                "family": args.family,
                "model": kind,
                "device": str(device),
                "steps": args.steps,
                "total_train_params": fit["params"],
                "active_params_per_step": fit["active_params"],
                "estimated_train_flops": fit["estimated_train_flops"],
                "training_seconds": fit["seconds"],
                # Validation accuracies
                "val_seen_pair_acc": eval_val_seen["example_accuracy"],
                "val_heldout_pair_acc": eval_val_heldout["example_accuracy"],
                # Test accuracies (Example-level)
                "seen_depth3_acc": seen_acc,
                "unseen_pair_depth3_acc": unseen_acc,
                "strict_functional_holdout_acc": strict_acc,
                "functional_leaked_pair_acc": leaked_acc,
                "depth4_extrapolation_acc": d4_acc,
                # Program Exact Match
                "seen_depth3_exact_prog_acc": eval_seen_d3["program_exact_accuracy"],
                "strict_functional_exact_prog_acc": eval_strict["program_exact_accuracy"],
                "functional_leaked_exact_prog_acc": eval_leaked["program_exact_accuracy"],
                "depth4_exact_prog_acc": eval_d4["program_exact_accuracy"],
                # Scientific Gaps
                "token_pair_generalization_gap": seen_acc - unseen_acc,
                "strict_functional_generalization_gap": seen_acc - strict_acc,
                "functional_leakage_advantage": leaked_acc - strict_acc,
                "split_stats": split.stats,
            }
            all_results.append(res)
            # Print single-line summary
            print(
                f"[{kind.upper():<12}] seen_d3={seen_acc:.3f} | "
                f"strict_holdout={strict_acc:.3f} | "
                f"leaked={leaked_acc:.3f} | "
                f"d4_extrap={d4_acc:.3f} | "
                f"params={fit['params']} | flops={fit['estimated_train_flops']:.2e}"
            )

    # Save outputs to run directory and root results
    (run_dir / "metrics.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    (ROOT / "results" / "latest_metrics.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    try:
        print(f"\nSaved {len(all_results)} experiment records to {run_dir / 'metrics.json'}")
    except UnicodeEncodeError:
        print(f"\nSaved {len(all_results)} experiment records to results/{run_dir.name}/metrics.json")

if __name__ == "__main__":
    main()
