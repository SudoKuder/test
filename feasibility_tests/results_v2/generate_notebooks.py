import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent

def make_demo_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# DreamerV3 Walker-Walk Demo & Smoke Test\n",
                    "This notebook demonstrates setup, training, and learning curve visualization on DMC `walker_walk` with proprioceptive inputs.\n",
                    "- Codebase: `NM512/dreamerv3-torch` (commit `6ef8646d807cd10ce0c88e10a7e943211e7fc44c`)\n",
                    "- Budget: 1,000 simulator frames (smoke test)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 1. Environment & Hardware Verification\n",
                    "import os, sys, torch\n",
                    "print('Python Version: ', sys.version.split()[0])\n",
                    "print('PyTorch Version:', torch.__version__)\n",
                    "print('CUDA Available: ', torch.cuda.is_available())\n",
                    "if torch.cuda.is_available():\n",
                    "    print('GPU Device:    ', torch.cuda.get_device_name(0))\n",
                    "device = 'cuda:0' if torch.cuda.is_available() else 'cpu'\n",
                    "print('Active Device: ', device)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 2. Run 1,000-Step Baseline Smoke Test\n",
                    "import subprocess\n",
                    "logdir = 'smoke_log_demo'\n",
                    "cmd = [\n",
                    "    sys.executable, 'dreamer.py',\n",
                    "    '--configs', 'dmc_proprio',\n",
                    "    '--task', 'dmc_walker_walk',\n",
                    "    '--logdir', logdir,\n",
                    "    '--steps', '1000',\n",
                    "    '--eval_every', '500',\n",
                    "    '--prefill', '250',\n",
                    "    '--eval_episode_num', '1',\n",
                    "    '--seed', '0',\n",
                    "    '--device', device\n",
                    "]\n",
                    "print('Executing:', ' '.join(cmd))\n",
                    "res = subprocess.run(cmd, capture_output=True, text=True)\n",
                    "print('Exit code:', res.returncode)\n",
                    "if res.returncode != 0:\n",
                    "    print('STDERR:\', res.stderr[-1500:])\n",
                    "    raise RuntimeError('Smoke test failed')\n",
                    "else:\n",
                    "    print('[SUCCESS] Smoke test completed successfully!')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 3. Parse and Plot Metrics\n",
                    "import json, pathlib\n",
                    "import matplotlib.pyplot as plt\n",
                    "\n",
                    "metrics_path = pathlib.Path(logdir) / 'metrics.jsonl'\n",
                    "steps, returns = [], []\n",
                    "if metrics_path.exists():\n",
                    "    with open(metrics_path) as f:\n",
                    "        for line in f:\n",
                    "            d = json.loads(line)\n",
                    "            if 'eval_return' in d:\n",
                    "                steps.append(d['step'])\n",
                    "                returns.append(d['eval_return'])\n",
                    "    print(f'Found {len(returns)} evaluation checkpoints:', list(zip(steps, returns)))\n",
                    "    plt.figure(figsize=(6, 3.5))\n",
                    "    plt.plot(steps, returns, marker='o', color='royalblue', label='Evaluation Return')\n",
                    "    plt.xlabel('Simulator Frames')\n",
                    "    plt.ylabel('Episodic Return')\n",
                    "    plt.title('DreamerV3 Walker-Walk Smoke Test')\n",
                    "    plt.grid(True, linestyle='--', alpha=0.6)\n",
                    "    plt.legend()\n",
                    "    plt.tight_layout()\n",
                    "    plt.show()\n",
                    "else:\n",
                    "    print('[WARN] No metrics.jsonl found.')"
                ]
            }
        ],
        "metadata": {
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open(HERE / "demo_notebook.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print("Created demo_notebook.ipynb")

def make_timing_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Target Hardware Benchmarking Notebook (Kaggle / Colab)\n",
                    "Measures seconds per 1,000 frames on target GPU hardware (e.g., Nvidia T4 / P100),\n",
                    "measures peak GPU memory and process RAM, and exports `timing_results.json`."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 1. Environment and Hardware Inspection\n",
                    "import os, sys, time, json, pathlib, subprocess, psutil, torch\n",
                    "gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'\n",
                    "print(f'Hardware: {gpu_name}')\n",
                    "print(f'PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 2. Benchmark Function\n",
                    "def run_timing_trial(steps, prefill, eval_every, label):\n",
                    "    logdir = f'timing_bench_{label}'\n",
                    "    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'\n",
                    "    cmd = [\n",
                    "        sys.executable, 'dreamer.py',\n",
                    "        '--configs', 'dmc_proprio',\n",
                    "        '--task', 'dmc_walker_walk',\n",
                    "        '--logdir', logdir,\n",
                    "        '--steps', str(steps),\n",
                    "        '--eval_every', str(eval_every),\n",
                    "        '--prefill', str(prefill),\n",
                    "        '--eval_episode_num', '1',\n",
                    "        '--seed', '0',\n",
                    "        '--device', device\n",
                    "    ]\n",
                    "    t0 = time.time()\n",
                    "    proc = subprocess.Popen(cmd)\n",
                    "    p = psutil.Process(proc.pid)\n",
                    "    peak_ram = 0.0\n",
                    "    peak_gpu = 0.0\n",
                    "    while proc.poll() is None:\n",
                    "        try:\n",
                    "            mem = p.memory_info().rss\n",
                    "            for c in p.children(recursive=True):\n",
                    "                mem += c.memory_info().rss\n",
                    "            peak_ram = max(peak_ram, mem / (1024 * 1024))\n",
                    "        except Exception:\n",
                    "            pass\n",
                    "        if torch.cuda.is_available():\n",
                    "            peak_gpu = max(peak_gpu, torch.cuda.max_memory_allocated() / (1024 * 1024))\n",
                    "        time.sleep(0.5)\n",
                    "    wall_sec = time.time() - t0\n",
                    "    sec_per_1k = (wall_sec / steps) * 1000.0\n",
                    "    print(f'[{label}] {steps} frames took {wall_sec:.1f}s ({sec_per_1k:.2f} s/1k frames) | Peak RAM: {peak_ram:.1f} MB | Peak GPU: {peak_gpu:.1f} MB')\n",
                    "    return {\n",
                    "        'frames': steps,\n",
                    "        'wall_clock_seconds': round(wall_sec, 2),\n",
                    "        'seconds_per_1000_frames': round(sec_per_1k, 2),\n",
                    "        'peak_ram_mb': round(peak_ram, 1),\n",
                    "        'peak_gpu_mb': round(peak_gpu, 1)\n",
                    "    }"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 3. Execute 2,000-Frame and 5,000-Frame Timing Trials\n",
                    "print('Starting 2,000-frame benchmark...')\n",
                    "trial_2k = run_timing_trial(2000, 500, 1000, 'trial_2k')\n",
                    "\n",
                    "print('Starting 5,000-frame benchmark...')\n",
                    "trial_5k = run_timing_trial(5000, 1000, 2000, 'trial_5k')\n",
                    "\n",
                    "# Compute estimated steady-state rate from the 5k trial\n",
                    "sec_per_1k_est = trial_5k['seconds_per_1000_frames']\n",
                    "est_50k_hours = (sec_per_1k_est * 50) / 3600.0\n",
                    "core_plan_6runs_hours = est_50k_hours * 6.0\n",
                    "\n",
                    "summary = {\n",
                    "    'hardware': gpu_name,\n",
                    "    'trial_2k': trial_2k,\n",
                    "    'trial_5k': trial_5k,\n",
                    "    'estimated_seconds_per_1000_frames': sec_per_1k_est,\n",
                    "    'extrapolated_50k_run_hours_ESTIMATE': round(est_50k_hours, 2),\n",
                    "    'extrapolated_core_plan_6runs_hours_ESTIMATE': round(core_plan_6runs_hours, 2),\n",
                    "    'fits_weekly_30h_quota': core_plan_6runs_hours <= 30.0\n",
                    "}\n",
                    "\n",
                    "with open('timing_results.json', 'w') as f:\n",
                    "    json.dump(summary, f, indent=2)\n",
                    "\n",
                    "print('\\n=== BENCHMARK SUMMARY ===')\n",
                    "print(json.dumps(summary, indent=2))\n",
                    "print('Saved timing_results.json')"
                ]
            }
        ],
        "metadata": {
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open(HERE / "timing_notebook.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print("Created timing_notebook.ipynb")

if __name__ == "__main__":
    make_demo_notebook()
    make_timing_notebook()
