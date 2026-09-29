#!/usr/bin/env python3
"""
compute_budget_v2.py  -  Upgraded Compute Budget Analysis

Takes measured per-run wall-clock times from all completed sequential seeds,
computes median and range, applies target hardware speed factors (e.g. Nvidia T4 / P100 vs RTX 4060),
and checks against Colab/Kaggle free-tier weekly GPU quotas (~30 GPU-hours/week).
"""
import argparse
import json
import pathlib
import sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

def main():
    parser = argparse.ArgumentParser(description="Upgraded Compute Budget Analysis v2")
    parser.add_argument("--gpu_speed_factor", type=float, default=1.0,
                        help="Relative execution time multiplier on target hardware (e.g., 1.0=RTX 4060, 1.35=Nvidia T4, 1.15=Nvidia P100)")
    parser.add_argument("--hardware_name", type=str, default="RTX 4060 Laptop GPU",
                        help="Name of target benchmark hardware")
    parser.add_argument("--weekly_quota", type=float, default=30.0,
                        help="Available GPU quota in hours (default: 30.0 h/week)")
    parser.add_argument("--out", type=str, default="compute_budget_v2.json",
                        help="Output JSON file name")
    args = parser.parse_args()

    # Load cleaned sequential runs
    cleaned_dir = HERE / "cleaned"
    runs = []
    
    # Check baseline seeds 0, 1, 2
    for s in [0, 1, 2]:
        p = cleaned_dir / f"baseline_seed{s}_cleaned.json"
        if p.exists():
            with open(p) as f:
                d = json.load(f)
                runs.append({
                    "seed": s,
                    "wall_clock_seconds": d["wall_clock_seconds"],
                    "hours": d["wall_clock_seconds"] / 3600.0,
                    "peak_gpu_mb": d["peak_gpu_mem_mb"],
                    "notes": d.get("notes", "")
                })

    if not runs:
        # Fallback to base dir
        print("[WARN] No cleaned files found, checking base directory...")
        for s in [1, 2]:
            p = ROOT / f"baseline_seed{s}.json"
            if p.exists():
                with open(p) as f:
                    d = json.load(f)
                    runs.append({
                        "seed": s,
                        "wall_clock_seconds": d["wall_clock_seconds"],
                        "hours": d["wall_clock_seconds"] / 3600.0,
                        "peak_gpu_mb": d["peak_gpu_mem_mb"]
                    })

    durations_s = [r["wall_clock_seconds"] for r in runs]
    durations_h = [r["hours"] for r in runs]
    peak_gpus = [r["peak_gpu_mb"] for r in runs]

    median_s = float(np.median(durations_s))
    median_h = median_s / 3600.0
    min_h = float(np.min(durations_h))
    max_h = float(np.max(durations_h))
    mean_peak_gpu = float(np.mean(peak_gpus))

    # Apply target hardware factor
    target_median_h = median_h * args.gpu_speed_factor
    target_min_h = min_h * args.gpu_speed_factor
    target_max_h = max_h * args.gpu_speed_factor

    # Plans
    # 1. Core Plan: Baseline (3 seeds) + Variant continuous latents (3 seeds) = 6 runs
    core_runs = 6
    core_hours_median = core_runs * target_median_h
    core_hours_range = [core_runs * target_min_h, core_runs * target_max_h]
    core_fits = core_hours_median <= args.weekly_quota

    # 2. Full Plan: 1 baseline + 3 variants + 2 ablations = 6 configs x 3 seeds = 18 runs
    full_runs = 18
    full_hours_median = full_runs * target_median_h
    full_hours_range = [full_runs * target_min_h, full_runs * target_max_h]
    full_fits = full_hours_median <= args.weekly_quota

    # Maximum 50k runs possible within weekly quota
    max_possible_runs = int(args.weekly_quota // target_median_h)

    summary = {
        "benchmark_hardware": args.hardware_name,
        "speed_factor_applied": args.gpu_speed_factor,
        "measured_seeds_count": len(runs),
        "per_seed_measured_hours": [round(h, 2) for h in durations_h],
        "per_run_duration_hours": {
            "median": round(target_median_h, 2),
            "min": round(target_min_h, 2),
            "max": round(target_max_h, 2),
            "unscaled_rtx4060_median_hours": round(median_h, 2)
        },
        "peak_gpu_mem_mb": round(mean_peak_gpu, 1),
        "weekly_quota_hours": args.weekly_quota,
        "core_plan_6_runs": {
            "configurations": "Baseline (3 seeds) + Continuous Latents (3 seeds)",
            "total_runs": core_runs,
            "estimated_gpu_hours": round(core_hours_median, 2),
            "gpu_hours_range": [round(core_hours_range[0], 2), round(core_hours_range[1], 2)],
            "fits_weekly_quota": core_fits
        },
        "full_plan_18_runs": {
            "configurations": "1 Baseline + 3 Variants + 2 Ablations (18 runs)",
            "total_runs": full_runs,
            "estimated_gpu_hours": round(full_hours_median, 2),
            "gpu_hours_range": [round(full_hours_range[0], 2), round(full_hours_range[1], 2)],
            "fits_weekly_quota": full_fits
        },
        "max_50k_runs_within_quota": max_possible_runs
    }

    out_file = HERE / args.out
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print("Compute Budget Feasibility Analysis v2")
    print(f"Target Hardware: {args.hardware_name} (Speed factor: {args.gpu_speed_factor}x)")
    print("=" * 80)
    print(f"Per-run 50k duration: median {target_median_h:.2f} h (range [{target_min_h:.2f}h, {target_max_h:.2f}h])")
    print(f"Peak GPU VRAM:       {mean_peak_gpu:.0f} MB")
    print(f"Weekly GPU Quota:    {args.weekly_quota:.1f} hours\n")
    print(f"Core Plan (6 runs) : {core_hours_median:.1f} hours (range [{core_hours_range[0]:.1f}h, {core_hours_range[1]:.1f}h]) -> Fits Quota: {'YES' if core_fits else 'NO'}")
    print(f"Full Plan (18 runs): {full_hours_median:.1f} hours (range [{full_hours_range[0]:.1f}h, {full_hours_range[1]:.1f}h]) -> Fits Quota: {'YES' if full_fits else 'NO'}")
    print(f"Max 50k runs fitting quota: {max_possible_runs} runs")
    print("=" * 80)
    print(f"Saved results to {out_file}")

if __name__ == "__main__":
    main()
