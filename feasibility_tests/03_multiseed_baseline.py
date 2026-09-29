#!/usr/bin/env python3
"""
03_multiseed_baseline.py

Runs the baseline DreamerV3 model across N seeds (default 3, configurable).
Extracts the pooled evaluation episodes (10 evaluation episodes per seed)
and computes robust statistical metrics:
- Interquartile Mean (IQM)
- Mean return
- Standard deviation
- Range (min and max)
- Per-seed return distributions

Saves results to multiseed_results.json.
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent

def calculate_iqm(returns):
    """
    Computes Interquartile Mean (IQM): mean of values between 25th and 75th percentiles.
    Standard metric for robust RL evaluation (Agarwal et al., 2021).
    """
    if not returns:
        return 0.0
    arr = np.array(returns, dtype=np.float64)
    if len(arr) < 4:
        return float(np.mean(arr))
    q25, q75 = np.percentile(arr, [25, 75])
    iqm_subset = arr[(arr >= q25) & (arr <= q75)]
    if len(iqm_subset) == 0:
        return float(np.mean(arr))
    return float(np.mean(iqm_subset))

def get_eval_returns(seed, num_eps=10):
    """Load rewards from the latest num_eps evaluation episodes in logdir."""
    eval_dir = HERE / f"logdir/walker_walk_seed{seed}" / "eval_eps"
    if not eval_dir.exists():
        return []
    
    files = list(eval_dir.glob("*.npz"))
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    latest_files = files[:num_eps]
    
    returns = []
    for f in latest_files:
        try:
            with np.load(f) as data:
                if 'reward' in data:
                    returns.append(float(np.sum(data['reward'])))
        except Exception:
            pass
    return returns

def main():
    parser = argparse.ArgumentParser(description="Multi-seed Baseline Evaluation")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds to run (default: 3)")
    parser.add_argument("--steps", type=int, default=50_000, help="Env step budget per run (default: 50,000)")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cuda:0 or cpu, default: auto-detect)")
    parser.add_argument("--force", action="store_true", help="Force re-run even if seed checkpoint exists")
    parser.add_argument("--out", type=str, default="multiseed_results.json", help="Output summary file")
    args = parser.parse_args()

    print("=" * 70)
    print(f"Starting Multi-seed Baseline Run: {args.seeds} seeds × {args.steps:,} steps")
    print("=" * 70)

    results_by_seed = {}
    pooled_returns = []

    for seed in range(args.seeds):
        existing_eps = get_eval_returns(seed, 10)
        seed_done = (HERE / f"logdir/walker_walk_seed{seed}" / "latest.pt").exists() and len(existing_eps) > 0

        if seed_done and not getattr(args, "force", False):
            print(f"\n>>> Seed {seed} ({seed + 1}/{args.seeds}) already completed. Reusing {len(existing_eps)} eval episodes. <<<")
            eps = existing_eps
        else:
            print(f"\n>>> Running Seed {seed} ({seed + 1}/{args.seeds}) <<<")
            cmd = [
                sys.executable, str(HERE / "01_baseline_run.py"),
                "--steps", str(args.steps),
                "--seed", str(seed),
                "--out", f"baseline_seed{seed}.json"
            ]
            if args.device:
                cmd.extend(["--device", args.device])
            subprocess.check_call(cmd, cwd=str(HERE))
            eps = get_eval_returns(seed, 10)

        results_by_seed[str(seed)] = eps
        pooled_returns.extend(eps)
        print(f"Seed {seed} evaluated returns ({len(eps)} episodes): {eps}")

    if pooled_returns:
        mean_ret = float(np.mean(pooled_returns))
        std_ret = float(np.std(pooled_returns))
        range_ret = [float(np.min(pooled_returns)), float(np.max(pooled_returns))]
        iqm_ret = calculate_iqm(pooled_returns)
    else:
        mean_ret = std_ret = iqm_ret = 0.0
        range_ret = [0.0, 0.0]

    summary = {
        "seeds_evaluated": args.seeds,
        "steps_per_seed": args.steps,
        "episodes_per_seed": 10,
        "total_pooled_episodes": len(pooled_returns),
        "per_seed_returns": results_by_seed,
        "pooled_mean": round(mean_ret, 2),
        "pooled_iqm": round(iqm_ret, 2),
        "pooled_std": round(std_ret, 2),
        "pooled_range": [round(range_ret[0], 2), round(range_ret[1], 2)]
    }
    
    out_file = HERE / args.out
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print("\n" + "=" * 70)
    print("MULTI-SEED STATISTICAL SUMMARY")
    print(json.dumps(summary, indent=2))
    print(f"Saved results to {out_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()
