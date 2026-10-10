import torch
import numpy as np
from ssnn.models import build_model
from ssnn.tasks import make_family, batchify
from ssnn.train import set_seed

def test_all_models_have_finite_outputs_and_correct_shapes():
    f = make_family("affine", 11)
    prog_rows = [((0, 1, 2), 3, int(f.run((0, 1, 2), [3])[0])) for _ in range(8)]
    prog, x, y, lengths = batchify(prog_rows, f.n_ops, f.n_states)
    
    kinds = ["mlp", "shared", "explicit", "library", "soft_library", "hypernet"]
    for kind in kinds:
        m = build_model(kind, f.n_ops, f.n_states, hidden=16)
        out = m(prog, x, lengths)
        assert out.shape == (8, f.n_states), f"Wrong shape for {kind}"
        assert torch.isfinite(out).all(), f"Non-finite output for {kind}"

def test_padding_and_variable_length_programs():
    f = make_family("affine", 11)
    # Batch containing programs of length 1, 2, 3, 4
    mixed_rows = [
        ((0,), 1, int(f.run((0,), [1])[0])),
        ((0, 1), 2, int(f.run((0, 1), [2])[0])),
        ((0, 1, 2), 3, int(f.run((0, 1, 2), [3])[0])),
        ((0, 1, 2, 3), 4, int(f.run((0, 1, 2, 3), [4])[0])),
    ]
    prog, x, y, lengths = batchify(mixed_rows, f.n_ops, f.n_states, max_depth=4)
    
    for kind in ["mlp", "shared", "explicit", "library", "soft_library", "hypernet"]:
        m = build_model(kind, f.n_ops, f.n_states, hidden=16, max_depth=4)
        out = m(prog, x, lengths)
        assert out.shape == (4, f.n_states)
        assert torch.isfinite(out).all()

def test_gradients_and_optimizer_step():
    f = make_family("affine", 11)
    rows = [((0, 1), 2, 5) for _ in range(4)]
    prog, x, y, lengths = batchify(rows, f.n_ops, f.n_states)
    
    for kind in ["mlp", "shared", "explicit", "library", "soft_library", "hypernet"]:
        m = build_model(kind, f.n_ops, f.n_states, hidden=16)
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        
        # Initial parameters
        p_initial = [p.clone() for p in m.parameters() if p.requires_grad]
        
        out = m(prog, x, lengths)
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()
        
        # Verify gradients exist and are finite
        for p in m.parameters():
            if p.requires_grad:
                assert p.grad is not None, f"Missing gradient for {kind}"
                assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {kind}"
                
        opt.step()
        # Verify weights were updated
        changed = any(not torch.equal(p1, p2) for p1, p2 in zip(p_initial, [p for p in m.parameters() if p.requires_grad]))
        assert changed, f"Weights did not update in {kind}"

def test_fixed_seed_reproducibility():
    f = make_family("affine", 11)
    rows = [((0, 1), 2, 5) for _ in range(4)]
    prog, x, y, lengths = batchify(rows, f.n_ops, f.n_states)
    
    set_seed(42)
    m1 = build_model("mlp", f.n_ops, f.n_states, hidden=16)
    out1 = m1(prog, x, lengths)
    
    set_seed(42)
    m2 = build_model("mlp", f.n_ops, f.n_states, hidden=16)
    out2 = m2(prog, x, lengths)
    
    assert torch.equal(out1, out2)
