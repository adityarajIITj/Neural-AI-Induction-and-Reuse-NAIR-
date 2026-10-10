import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def analyze_run(run_dir):
    metrics_file = run_dir / "metrics.json"
    data = json.loads(metrics_file.read_text(encoding="utf-8"))
    
    family = data[0]["family"]
    models = sorted(list({d["model"] for d in data}))
    
    print(f"\n==================== Family: {family.upper()} ====================")
    header = (
        f"{'Model':<14} | {'Params':<8} | {'Active':<8} | {'FLOPs':<9} | "
        f"{'Seen D3':<13} | {'Strict Holdout':<14} | {'Leaked D3':<13} | "
        f"{'Leak Gap':<10} | {'D4 Extrap':<13}"
    )
    print(header)
    print("-" * len(header))
    
    for m in models:
        m_data = [d for d in data if d["model"] == m]
        params = m_data[0]["total_train_params"]
        active = m_data[0]["active_params_per_step"]
        flops = m_data[0]["estimated_train_flops"]
        
        seen = [d["seen_depth3_acc"] for d in m_data]
        strict = [d["strict_functional_holdout_acc"] for d in m_data]
        leaked = [d["functional_leaked_pair_acc"] for d in m_data]
        leak_gap = [d["functional_leakage_advantage"] for d in m_data]
        d4 = [d["depth4_extrapolation_acc"] for d in m_data]
        
        def fmt(vals):
            return f"{np.mean(vals)*100:5.1f}% +/-{np.std(vals)*100:4.1f}%"
            
        print(
            f"{m.upper():<14} | {params:<8} | {active:<8} | {flops:.2e} | "
            f"{fmt(seen):<13} | {fmt(strict):<14} | {fmt(leaked):<13} | "
            f"{np.mean(leak_gap)*100:+5.1f}%     | {fmt(d4):<13}"
        )

def main():
    results_dir = ROOT / "results"
    run_dirs = sorted([d for d in results_dir.glob("pilot_*") if d.is_dir()])
    for rd in run_dirs:
        analyze_run(rd)

if __name__ == "__main__":
    main()
