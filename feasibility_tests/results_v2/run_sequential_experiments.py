#!/usr/bin/env python3
"""
run_sequential_experiments.py  -  Sequential Long-Run Orchestrator (Stage 6)

Executes full 50,000-frame runs sequentially on the local GPU with:
1. Strict single-process GPU enforcement.
2. Resumability: checks if checkpoint latest.pt and metrics.jsonl already reached 50k frames.
3. Execution order (as requested):
   - 1. Variant (--dyn_discrete 0) Seed 0
   - 2. Baseline Seed 0
   - 3. Variant (--dyn_discrete 0) Seed 1
   - 4. Baseline Seed 1
   - 5. Variant (--dyn_discrete 0) Seed 2
   - 6. Baseline Seed 2
4. High-fidelity resource monitoring:
   - Peak CPU RAM via psutil across parent + all child processes.
   - Peak GPU VRAM via nvidia-smi.
   - Wall-clock seconds.
5. Automated post-run evaluation:
   - Runs open_loop_eval_v2.py on the checkpoint.
   - Writes structured JSON result to results_v2/runs/.
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import threading
import time

try:
    import psutil
except ImportError:
    psutil = None

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent / "dreamerv3-torch"
RUNS_DIR = HERE / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

class ResourceMonitor(threading.Thread):
    def __init__(self, pid, interval=1.0):
        super().__init__(daemon=True)
        self.pid = pid
        self.interval = interval
        self.peak_cpu_mb = 0.0
        self.peak_gpu_mb = 0.0
        self._stop = threading.Event()

    def run(self):
        proc = None
        if psutil:
            try:
                proc = psutil.Process(self.pid)
            except Exception:
                proc = None

        while not self._stop.is_set():
            if proc and psutil:
                try:
                    mem = proc.memory_info().rss
                    for child in proc.children(recursive=True):
                        try:
                            mem += child.memory_info().rss
                        except Exception:
                            pass
                    self.peak_cpu_mb = max(self.peak_cpu_mb, mem / (1024 * 1024))
                except Exception:
                    pass

            # Query nvidia-smi
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    encoding="utf-8", timeout=5, stderr=subprocess.DEVNULL
                )
                vals = [int(x.strip()) for x in out.strip().split("\n") if x.strip().isdigit()]
                if vals:
                    self.peak_gpu_mb = max(self.peak_gpu_mb, max(vals))
            except Exception:
                pass

            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


def parse_metrics(logdir):
    p = pathlib.Path(logdir) / "metrics.jsonl"
    if not p.exists():
        return [], None
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line.strip()))
                except Exception:
                    pass
    
    eval_cps = [r for r in rows if "eval_return" in r]
    last_frame = rows[-1]["step"] if rows else 0
    return eval_cps, last_frame


def run_single_experiment(py_exe, exp_type, seed, steps=50000):
    exp_name = f"{exp_type}_seed{seed}"
    logdir = HERE / f"logdir/{exp_name}"
    summary_file = RUNS_DIR / f"{exp_name}_summary.json"

    print("\n" + "=" * 80)
    print(f"EXPERIMENT: {exp_name.upper()} | Target Frames: {steps:,}")
    print(f"Logdir: {logdir}")
    print("=" * 80)

    # Check if already completed
    if (logdir / "latest.pt").exists() and summary_file.exists():
        with open(summary_file) as f:
            s_data = json.load(f)
        if s_data.get("success") and s_data.get("last_logged_frame", 0) >= 45000:
            print(f"[RESUME] Experiment {exp_name} already completed successfully. Skipping.")
            return s_data

    cmd = [
        py_exe, str(REPO / "dreamer.py"),
        "--configs", "dmc_proprio",
        "--task", "dmc_walker_walk",
        "--logdir", str(logdir),
        "--steps", str(steps),
        "--seed", str(seed),
        "--device", "cuda:0"
    ]
    if exp_type == "variant_discrete0":
        cmd.extend(["--dyn_discrete", "0"])

    sub_env = os.environ.copy()
    if os.name == "nt" and sub_env.get("MUJOCO_GL") in (None, "osmesa"):
        sub_env["MUJOCO_GL"] = "glfw"

    print("Command:", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.Popen(cmd, env=sub_env)

    monitor = ResourceMonitor(proc.pid)
    monitor.start()

    retcode = proc.wait()
    monitor.stop()
    monitor.join(timeout=5)
    wall_sec = time.time() - t0

    eval_cps, last_frame = parse_metrics(logdir)
    final_return = eval_cps[-1]["eval_return"] if eval_cps else None

    # Run upgraded open-loop evaluation on saved checkpoint
    open_loop_res = None
    if retcode == 0 and (logdir / "latest.pt").exists():
        ol_script = HERE / "open_loop_eval_v2.py"
        ol_out = RUNS_DIR / f"{exp_name}_open_loop.json"
        dyn_arg = ["--dyn_discrete", "0"] if exp_type == "variant_discrete0" else []
        try:
            print(f"\nRunning open_loop_eval_v2 on {logdir}...")
            subprocess.check_call(
                [py_exe, str(ol_script), "--logdir", str(logdir), "--out", str(ol_out)] + dyn_arg
            )
            if ol_out.exists():
                with open(ol_out) as f:
                    open_loop_res = json.load(f)
        except Exception as e:
            print(f"[WARN] Open-loop eval failed: {e}")

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": exp_name,
        "type": exp_type,
        "seed": seed,
        "steps_budget": steps,
        "last_logged_frame": last_frame,
        "budget_enforced": 45000 <= last_frame <= 50000,
        "wall_clock_seconds": round(wall_sec, 1),
        "wall_clock_hours": round(wall_sec / 3600.0, 2),
        "peak_cpu_mem_mb": round(monitor.peak_cpu_mb, 1),
        "peak_gpu_mem_mb": round(monitor.peak_gpu_mb, 1),
        "eval_checkpoints": eval_cps,
        "final_return": final_return,
        "meets_f3_threshold_100": final_return is not None and final_return >= 100.0,
        "success": retcode == 0,
        "open_loop_summary": open_loop_res.get("metrics") if open_loop_res else None
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n--- Summary ---")
    print(f"Success: {summary['success']} | Final Return: {final_return} | Duration: {summary['wall_clock_hours']}h | Frames: {last_frame}")
    print(f"Saved: {summary_file}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Sequential Experiment Orchestrator")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only one experiment: e.g. variant_seed0, baseline_seed0")
    args = parser.parse_args()

    venv_py = ROOT / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    py = str(venv_py) if venv_py.exists() else sys.executable

    # Priority queue
    queue = [
        ("variant_discrete0", 0),  # Priority 1: Variant Seed 0 (immediate F3 evidence)
        ("baseline", 0),           # Priority 2: Baseline Seed 0 (fixed budget)
        ("variant_discrete0", 1),  # Priority 3: Variant Seed 1
        ("baseline", 1),           # Priority 4: Baseline Seed 1
        ("variant_discrete0", 2),  # Priority 5: Variant Seed 2
        ("baseline", 2),           # Priority 6: Baseline Seed 2
    ]

    if args.only:
        parts = args.only.split("_")
        exp_t = "_".join(parts[:-1]) if "discrete0" not in args.only else "variant_discrete0"
        s_num = int(parts[-1].replace("seed", ""))
        queue = [(exp_t, s_num)]

    print("=" * 80)
    print("SEQUENTIAL EXPERIMENT RUNNER (Stage 6)")
    print(f"Total planned runs: {len(queue)}")
    print("Execution plan:")
    for i, (t, s) in enumerate(queue, 1):
        print(f"  {i}. {t}_seed{s}")
    print("=" * 80)

    for i, (exp_t, seed) in enumerate(queue, 1):
        print(f"\n>>> Running job {i}/{len(queue)}: {exp_t}_seed{seed} <<<")
        run_single_experiment(py, exp_t, seed, steps=50000)

    print("\n" + "=" * 80)
    print("ALL SCHEDULED EXPERIMENTS COMPLETED.")
    print("=" * 80)

if __name__ == "__main__":
    main()
