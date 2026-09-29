#!/usr/bin/env python3
"""
06_compute_budget.py

Computes total GPU-hours needed for the entire planned experimental campaign:
- Baseline (1 configuration)
- World Model Variants (default 3: e.g. Continuous latents, Recurrent depth, etc.)
- Ablations (default 2: e.g. KL balancing scale, Latent dimensionality)
- Seeds per configuration (default 3)

Formula:
Total Runs = (1 + num_variants + num_ablations) * num_seeds
Total GPU Hours = Total Runs * (Wall-Clock Seconds Per 50k Run / 3600)

Accepts CLI flags:
--available_gpu_hours (default 50.0)
--num_variants (default 3)
--num_ablations (default 2)
--num_seeds (default 3)
--measured_seconds (optional override if baseline_results.json not present)
"""
import argparse
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent

def get_measured_time():
    b_file = HERE / "baseline_results.json"
    if b_file.exists():
        try:
            with open(b_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("success") and data.get("wall_clock_seconds", 0) > 10:
                    secs = data.get("wall_clock_seconds")
                    return float(secs), f"Measured from baseline_results.json ({data.get('steps')} steps)"
        except Exception:
            pass
    return None, "No completed baseline run found (using conservative estimate)"

def main():
    parser = argparse.ArgumentParser(description="Compute Budget Feasibility Analysis")
    parser.add_argument("--available_gpu_hours", type=float, default=50.0,
                        help="Available GPU hours (e.g. Colab Pro / Kaggle weekly quota, default 50.0)")
    parser.add_argument("--num_variants", type=int, default=3, help="Number of planned world model variants (default: 3)")
    parser.add_argument("--num_ablations", type=int, default=2, help="Number of planned ablations (default: 2)")
    parser.add_argument("--num_seeds", type=int, default=3, help="Seeds per configuration (default: 3)")
    parser.add_argument("--measured_seconds", type=float, default=None,
                        help="Manual wall-clock seconds per 50k run (optional)")
    args = parser.parse_args()

    print("=" * 70)
    print("06: Compute Budget Feasibility Analysis")
    print("=" * 70)

    # Determine runtime per run
    measured_sec, source = get_measured_time()
    if args.measured_seconds is not None:
        sec_per_run = args.measured_seconds
        source = f"Manually supplied (--measured_seconds={sec_per_run})"
    elif measured_sec is not None:
        sec_per_run = measured_sec
    else:
        # Fallback conservative estimate (~35 min on T4 / RTX 4060 for 50k steps)
        sec_per_run = 2100.0
        source = "Estimated default (2100s / 35 min per 50k run on modern GPU)"

    hours_per_run = sec_per_run / 3600.0
    configs = 1 + args.num_variants + args.num_ablations
    total_runs = configs * args.num_seeds
    total_gpu_hours = total_runs * hours_per_run

    print(f"\n[Run Parameters]")
    print(f"- Wall-clock time per 50k run: {sec_per_run:.1f}s ({hours_per_run:.2f} hours)")
    print(f"  Source: {source}")
    print(f"- Experimental configurations: 1 baseline + {args.num_variants} variants + {args.num_ablations} ablations = {configs} configs")
    print(f"- Seeds per config: {args.num_seeds}")
    print(f"- Total training runs: {total_runs}")
    print(f"- Available GPU quota: {args.available_gpu_hours:.1f} GPU-hours")
    print(f"- Total GPU hours required: {total_gpu_hours:.2f} GPU-hours")

    fits = total_gpu_hours <= args.available_gpu_hours

    print("\n" + "=" * 70)
    if fits:
        margin = args.available_gpu_hours - total_gpu_hours
        print(f"VERDICT: [FEASIBLE] Plan fits within GPU budget with {margin:.1f} hours margin.")
    else:
        excess = total_gpu_hours - args.available_gpu_hours
        print(f"VERDICT: [EXCEEDS BUDGET] Plan exceeds GPU budget by {excess:.1f} GPU-hours.")
        print("\n[Actionable Pruning Recommendations]")
        max_possible_runs = int(args.available_gpu_hours / hours_per_run)
        print(f"1. Maximum full 50k runs possible with {args.available_gpu_hours}h: {max_possible_runs} runs.")
        
        # Pruning strategies:
        # Option A: Cut ablations
        no_ablation_runs = (1 + args.num_variants) * args.num_seeds
        no_ablation_hours = no_ablation_runs * hours_per_run
        print(f"2. Cut all {args.num_ablations} ablations -> {no_ablation_runs} runs = {no_ablation_hours:.1f}h (Fits? {'YES' if no_ablation_hours <= args.available_gpu_hours else 'NO'})")
        
        # Option B: Cut ablations + reduce to 1 variant
        min_runs = (1 + 1) * args.num_seeds
        min_hours = min_runs * hours_per_run
        print(f"3. Core plan (Baseline + 1 Variant × {args.num_seeds} seeds) -> {min_runs} runs = {min_hours:.1f}h (Fits? {'YES' if min_hours <= args.available_gpu_hours else 'NO'})")
    print("=" * 70 + "\n")

    report_data = {
        "wall_clock_seconds_per_run": sec_per_run,
        "hours_per_run": round(hours_per_run, 3),
        "source": source,
        "configurations": {
            "baseline": 1,
            "variants": args.num_variants,
            "ablations": args.num_ablations,
            "total_configs": configs
        },
        "seeds_per_config": args.num_seeds,
        "total_runs": total_runs,
        "required_gpu_hours": round(total_gpu_hours, 2),
        "available_gpu_hours": args.available_gpu_hours,
        "fits": fits
    }

    out_file = HERE / "compute_budget_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"Saved compute budget report to {out_file}")

if __name__ == "__main__":
    main()
