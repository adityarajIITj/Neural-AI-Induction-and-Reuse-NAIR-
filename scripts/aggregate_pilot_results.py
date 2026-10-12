import argparse
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

    summary_records = {}

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

        summary_records[m] = {
            "params": params,
            "active_params": active,
            "flops": flops,
            "seen_mean": float(np.mean(seen)),
            "strict_mean": float(np.mean(strict)),
            "leaked_mean": float(np.mean(leaked)),
            "d4_mean": float(np.mean(d4)),
        }

    return {family: summary_records}

def main():
    ap = argparse.ArgumentParser(description="Aggregate and summarize pilot benchmark results")
    ap.add_argument("--save-summary", action="store_true", help="Save aggregated metrics summary to results/summary.json")
    args = ap.parse_args()

    results_dir = ROOT / "results"
    run_dirs = sorted([d for d in results_dir.glob("pilot_*") if d.is_dir()])
    all_summaries = {}
    for rd in run_dirs:
        res = analyze_run(rd)
        all_summaries.update(res)

    if args.save_summary:
        out_path = results_dir / "summary.json"
        out_path.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
        print(f"\nSaved summary to {out_path.name}")

if __name__ == "__main__":
    main()
