#!/usr/bin/env python3
"""
01_baseline_run.py  –  Run the unmodified dreamerv3-torch on walker_walk
                       (dmc_proprio) for a given number of steps (default 50 000).

Logs wall-clock time, peak GPU / CPU memory, eval returns from metrics.jsonl,
and saves a JSON summary to baseline_results.json.

Usage (from feasibility_tests/):
    # Smoke test
    python 01_baseline_run.py --steps 2000

    # Full 50k run
    python 01_baseline_run.py --steps 50000 --seed 0
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
import threading

REPO = pathlib.Path(__file__).resolve().parent.parent / "dreamerv3-torch"
HERE = pathlib.Path(__file__).resolve().parent

def get_python_exe():
    venv = HERE / "venv"
    if os.name == "nt":
        venv_py = venv / "Scripts" / "python.exe"
    else:
        venv_py = venv / "bin" / "python"
    
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def parse_metrics_jsonl(logdir):
    """Read metrics.jsonl and return list of dicts."""
    p = pathlib.Path(logdir) / "metrics.jsonl"
    if not p.exists():
        return []
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


class PeakMemMonitor(threading.Thread):
    """Background thread that polls peak CPU RSS (via psutil) and GPU mem."""
    def __init__(self, pid, interval=2.0):
        super().__init__(daemon=True)
        self.pid = pid
        self.interval = interval
        self.peak_cpu_mb = 0.0
        self.peak_gpu_mb = 0.0
        self._stop = threading.Event()

    def run(self):
        try:
            import psutil
        except ImportError:
            psutil = None
        
        proc = None
        if psutil:
            try:
                proc = psutil.Process(self.pid)
            except Exception:
                proc = None

        while not self._stop.is_set():
            if proc:
                try:
                    mem = proc.memory_info().rss
                    for child in proc.children(recursive=True):
                        try:
                            mem += child.memory_info().rss
                        except Exception:
                            pass
                    mem_mb = mem / (1024 * 1024)
                    self.peak_cpu_mb = max(self.peak_cpu_mb, mem_mb)
                except Exception:
                    break

            # GPU via nvidia-smi (lightweight query)
            try:
                out = subprocess.check_output(
                    ["nvidia-smi",
                     "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    encoding="utf-8", timeout=5, stderr=subprocess.DEVNULL
                )
                gpu_mb = max(int(x.strip()) for x in out.strip().split("\n") if x.strip().isdigit())
                self.peak_gpu_mb = max(self.peak_gpu_mb, gpu_mb)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


def get_device(py_exe, explicit_device=None):
    if explicit_device:
        return explicit_device
    try:
        out = subprocess.check_output([py_exe, "-c", "import torch; print(torch.cuda.is_available())"], text=True, stderr=subprocess.DEVNULL).strip()
        if out == "True":
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Run Baseline DreamerV3 on Walker-Walk")
    parser.add_argument("--steps", type=int, default=50_000,
                        help="Total env-step budget (default: 50,000)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cuda:0 or cpu, default: auto-detect)")
    parser.add_argument("--out", type=str, default="baseline_results.json", help="Output summary file")
    args = parser.parse_args()

    py = get_python_exe()
    device = get_device(py, args.device)
    logdir = HERE / f"logdir/walker_walk_seed{args.seed}"

    cmd = [
        py, str(REPO / "dreamer.py"),
        "--configs", "dmc_proprio",
        "--task", "dmc_walker_walk",
        "--logdir", str(logdir),
        "--steps", str(args.steps),
        "--seed", str(args.seed),
        "--device", device,
    ]
    print("=" * 70)
    print(f"Starting Baseline Run: Seed {args.seed} | Steps: {args.steps:,}")
    print(f"Python: {py}")
    print("Command:", " ".join(cmd))
    sub_env = os.environ.copy()
    if os.name == "nt" and sub_env.get("MUJOCO_GL") in (None, "osmesa"):
        sub_env["MUJOCO_GL"] = "glfw"

    t0 = time.time()
    proc = subprocess.Popen(cmd, env=sub_env)

    monitor = PeakMemMonitor(proc.pid)
    monitor.start()

    proc.wait()
    monitor.stop()
    monitor.join(timeout=5)

    wall_sec = time.time() - t0
    success = proc.returncode == 0

    # Parse logged eval returns
    metrics = parse_metrics_jsonl(logdir)
    eval_returns = [
        {"step": m["step"], "eval_return": m["eval_return"]}
        for m in metrics if "eval_return" in m
    ]
    final_return = eval_returns[-1]["eval_return"] if eval_returns else None

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "steps": args.steps,
        "seed": args.seed,
        "wall_clock_seconds": round(wall_sec, 1),
        "peak_cpu_mem_mb": round(monitor.peak_cpu_mb, 1),
        "peak_gpu_mem_mb": round(monitor.peak_gpu_mb, 1),
        "eval_checkpoints": eval_returns,
        "final_return": final_return,
        "success": success,
    }
    
    out = HERE / args.out
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 70)
    print("BASELINE RUN FINISHED")
    print(json.dumps(summary, indent=2))
    print(f"Saved summary to {out}")
    print("=" * 70)

if __name__ == "__main__":
    main()
