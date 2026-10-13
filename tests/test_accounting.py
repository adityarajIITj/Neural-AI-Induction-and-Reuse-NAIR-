from ssnn.accounting import count_params, count_active_params, estimate_train_flops, estimate_forward_flops
from ssnn.models import build_model

def test_parameter_count_and_positive_cost():
    for kind in ["mlp", "shared", "explicit", "library", "soft_library", "hypernet"]:
        m = build_model(kind, 12, 11, hidden=16)
        total_p = count_params(m)
        active_p = count_active_params(m, kind)
        assert total_p > 0, f"Zero parameters for {kind}"
        assert active_p > 0, f"Zero active parameters for {kind}"
        assert active_p <= total_p, f"Active parameters exceed total in {kind}"
        
        fwd_flops = estimate_forward_flops(m, batch_size=16, task_depth=3, kind=kind)
        train_flops = estimate_train_flops(m, batch_size=16, steps=10, task_depth=3, kind=kind)
        assert fwd_flops > 0, f"Zero forward FLOPs for {kind}"
        assert train_flops > 0, f"Zero train FLOPs for {kind}"
        # 1 training step ≈ 3x forward FLOPs
        assert train_flops == 10 * 3 * fwd_flops

def test_active_parameters_distinction_for_explicit():
    # Explicit modular model has n_ops units, but only 1 active unit per step
    m_exp = build_model("explicit", 12, 11, hidden=16)
    total_p = count_params(m_exp)
    active_p = count_active_params(m_exp, "explicit")
    assert total_p > active_p, "Explicit modular model should have active_params < total_params"

def test_inference_flops_matches_forward_flops():
    from ssnn.accounting import estimate_inference_flops
    m = build_model("shared", 12, 11, hidden=16)
    fwd = estimate_forward_flops(m, batch_size=8, task_depth=4, kind="shared")
    inf = estimate_inference_flops(m, batch_size=8, task_depth=4, kind="shared")
    assert fwd == inf
    assert inf > 0

def test_count_trainable_vs_frozen_params():
    from ssnn.accounting import count_trainable_vs_frozen_params
    m = build_model("mlp", 12, 11, hidden=16)
    trainable_init, frozen_init = count_trainable_vs_frozen_params(m)
    assert trainable_init > 0
    assert frozen_init == 0

    for p in m.parameters():
        p.requires_grad = False
        break

    trainable_post, frozen_post = count_trainable_vs_frozen_params(m)
    assert frozen_post > 0
    assert trainable_post + frozen_post == trainable_init
