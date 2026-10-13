from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class MonolithicMLP(nn.Module):
    """Fixed-capacity monolithic baseline receiving full program encoding and input."""
    def __init__(self, n_ops: int, n_states: int, max_depth: int = 4, hidden: int = 128):
        super().__init__()
        self.n_ops = n_ops
        self.n_states = n_states
        self.max_depth = max_depth
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(n_ops * max_depth + n_states, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_states),
        )

    def forward(self, prog: torch.Tensor, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        B, L = prog.shape
        # Distinguish real operation tokens from padding (PAD token == n_ops)
        is_active = (prog < self.n_ops).unsqueeze(-1)
        p = F.one_hot(prog.clamp(0, self.n_ops - 1), self.n_ops).float()
        # Padded slots get zero vectors
        p = torch.where(is_active, p, torch.zeros_like(p))

        # Adjust to self.max_depth if variable length
        if L < self.max_depth:
            pad_size = self.max_depth - L
            p = torch.cat([p, torch.zeros(B, pad_size, self.n_ops, device=prog.device, dtype=p.dtype)], dim=1)
        elif L > self.max_depth:
            p = p[:, :self.max_depth, :]

        p = p.reshape(B, -1)
        xi = F.one_hot(x, self.n_states).float()
        return self.net(torch.cat([p, xi], dim=-1))


class OpUnit(nn.Module):
    """Standard 2-layer MLP unit for state transformation."""
    def __init__(self, input_dim: int, output_dim: int, hidden: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Apply 2-layer MLP non-linear mapping to input state vector."""
        return self.net(state)

    def __repr__(self) -> str:
        return f"OpUnit(input_dim={self.input_dim}, output_dim={self.output_dim}, hidden={self.hidden})"


class HyperNetStep(nn.Module):
    """Prior-art baseline: HyperNetwork-generated dynamic linear step (Ha et al. 2016).

    An operation embedding generates the weight matrix for the step transformation.
    """
    def __init__(self, n_ops: int, n_states: int, hidden: int = 32, embed_dim: int = 16):
        super().__init__()
        self.n_ops = n_ops
        self.n_states = n_states
        self.hidden = hidden
        self.op_embed = nn.Embedding(n_ops + 1, embed_dim)
        # Generates W1 [n_states -> hidden], b1 [hidden], W2 [hidden -> n_states], b2 [n_states]
        w1_size = n_states * hidden
        w2_size = hidden * n_states
        self.gen_w1 = nn.Linear(embed_dim, w1_size)
        self.gen_b1 = nn.Linear(embed_dim, hidden)
        self.gen_w2 = nn.Linear(embed_dim, w2_size)
        self.gen_b2 = nn.Linear(embed_dim, n_states)

    def forward(self, state: torch.Tensor, op: torch.Tensor) -> torch.Tensor:
        B = state.shape[0]
        emb = self.op_embed(op.clamp(0, self.n_ops))  # [B, embed_dim]
        w1 = self.gen_w1(emb).reshape(B, self.n_states, self.hidden)
        b1 = self.gen_b1(emb).unsqueeze(1)
        w2 = self.gen_w2(emb).reshape(B, self.hidden, self.n_states)
        b2 = self.gen_b2(emb).unsqueeze(1)

        # Batch matrix multiplication: state [B, 1, n_states] @ w1 [B, n_states, hidden]
        h = torch.tanh(torch.bmm(state.unsqueeze(1), w1) + b1)
        out = torch.bmm(h, w2) + b2
        return out.squeeze(1)


class StepModel(nn.Module):
    """Step-wise composition model supporting:

    - 'shared': one single unit reused across all steps
    - 'explicit': operation-specific dedicated units
    - 'library': bank of reusable units with discrete straight-through routing
    - 'soft_library': continuous mixture-of-experts routing (Rosenbaum et al. 2018)
    - 'hypernet': dynamic token-conditioned hypernetwork transformation (Ha et al. 2016)
    """
    def __init__(
        self,
        n_ops: int,
        n_states: int,
        hidden: int = 64,
        mode: str = "shared",
        n_units: int | None = None,
        router: str = "st",
    ):
        super().__init__()
        self.n_ops = n_ops
        self.n_states = n_states
        self.mode = mode
        self.router = router
        self.hidden = hidden

        if mode == "hypernet":
            self.hyper_step = HyperNetStep(n_ops, n_states, hidden=hidden // 2)
            self.units = None
            self.router_logits = None
            return

        m = n_units or n_ops
        self.n_units = m

        if mode == "shared":
            self.units = nn.ModuleList([OpUnit(n_states + n_ops, n_states, hidden)])
            self.router_logits = None
        elif mode == "explicit":
            # Exactly n_ops units: unit i corresponds to operation i
            self.units = nn.ModuleList([OpUnit(n_states + n_ops, n_states, hidden) for _ in range(n_ops)])
            self.router_logits = None
        elif mode in ("library", "soft_library"):
            self.units = nn.ModuleList([OpUnit(n_states + n_ops, n_states, hidden) for _ in range(m)])
            # Learnable operation-to-unit router logits
            self.router_logits = nn.Parameter(torch.zeros(n_ops, m))
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def step(self, state: torch.Tensor, op: torch.Tensor) -> torch.Tensor:
        if self.mode == "hypernet":
            return self.hyper_step(state, op)

        # One-hot op representation
        is_active = (op < self.n_ops).unsqueeze(-1)
        op_hot = F.one_hot(op.clamp(0, self.n_ops - 1), self.n_ops).float()
        op_hot = torch.where(is_active, op_hot, torch.zeros_like(op_hot))
        z = torch.cat([state, op_hot], dim=-1)

        if self.mode == "shared":
            return self.units[0](z)

        outs = torch.stack([u(z) for u in self.units], dim=1)  # [B, M, S]

        if self.mode == "explicit":
            idx = op.clamp(0, len(self.units) - 1)
            return outs[torch.arange(len(op), device=op.device), idx]

        # Library routing
        probs = self.router_logits[op.clamp(0, self.n_ops - 1)].softmax(-1)
        if self.mode == "library" and self.router == "st":
            hard = F.one_hot(probs.argmax(-1), probs.shape[-1]).float()
            probs = hard - probs.detach() + probs
        return (outs * probs.unsqueeze(-1)).sum(1)

    def forward(self, prog: torch.Tensor, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        state = F.one_hot(x, self.n_states).float()
        for t in range(prog.shape[1]):
            op = prog[:, t]
            active = op < self.n_ops
            if not active.any():
                continue
            updated = self.step(state, op)
            state = torch.where(active[:, None], updated, state)
        return state


def build_model(
    kind: str,
    n_ops: int,
    n_states: int,
    hidden: int = 64,
    max_depth: int = 4,
    n_units: int | None = None,
) -> nn.Module:
    if kind == "mlp":
        return MonolithicMLP(n_ops, n_states, max_depth=max_depth, hidden=hidden)
    elif kind == "shared":
        return StepModel(n_ops, n_states, hidden=hidden, mode="shared")
    elif kind == "explicit":
        return StepModel(n_ops, n_states, hidden=hidden, mode="explicit")
    elif kind == "library":
        m = n_units or max(2, n_ops // 2)
        return StepModel(n_ops, n_states, hidden=hidden, mode="library", n_units=m, router="st")
    elif kind == "soft_library":
        m = n_units or max(2, n_ops // 2)
        return StepModel(n_ops, n_states, hidden=hidden, mode="soft_library", n_units=m, router="soft")
    elif kind == "hypernet":
        return StepModel(n_ops, n_states, hidden=hidden, mode="hypernet")
    else:
        raise ValueError(f"Unknown model kind: {kind}")


def get_model_summary(model: nn.Module) -> dict:
    """Return dictionary summary of model parameters, class name, and architecture mode."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mode = getattr(model, "mode", "monolithic" if isinstance(model, MonolithicMLP) else "unknown")
    return {
        "class_name": model.__class__.__name__,
        "mode": mode,
        "total_trainable_params": total_params,
        "n_ops": getattr(model, "n_ops", None),
        "n_states": getattr(model, "n_states", None),
    }
