import torch
from ssnn.models import build_model
from ssnn.tasks import make_family, sample_examples
from ssnn.train import train, evaluate_detailed

def test_train_reduces_loss_and_records_endpoints():
    device = torch.device("cpu")
    f = make_family("affine", 11)
    # Primitive programs (depth 1)
    primitive_programs = [(i,) for i in range(f.n_ops)]
    rows = sample_examples(f, primitive_programs, per_program=16, seed=42)

    model = build_model("shared", f.n_ops, f.n_states, hidden=32)
    fit = train(model, rows, f, device, steps=80, batch_size=32, lr=5e-3, seed=42)

    assert "initial_loss" in fit
    assert "final_loss" in fit
    assert fit["initial_loss"] is not None
    assert fit["final_loss"] is not None
    assert fit["final_loss"] < fit["initial_loss"], "Training loss did not decrease on primitives"

def test_evaluate_detailed_metrics_structure():
    device = torch.device("cpu")
    f = make_family("affine", 11)
    rows = sample_examples(f, [(0,), (1,)], per_program=8, seed=1)

    model = build_model("shared", f.n_ops, f.n_states, hidden=16)
    res = evaluate_detailed(model, rows, f, device)

    assert "example_accuracy" in res
    assert "program_exact_accuracy" in res
    assert "total_examples" in res
    assert "total_programs" in res
    assert 0.0 <= res["example_accuracy"] <= 1.0
    assert 0.0 <= res["program_exact_accuracy"] <= 1.0
