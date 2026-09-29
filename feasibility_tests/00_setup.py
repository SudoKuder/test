#!/usr/bin/env python3
"""
00_setup.py  –  Create a virtual environment, install pinned deps, write versions.txt.

Run from inside feasibility_tests/:
    python 00_setup.py

Requires: Python 3.11 (as specified by the dreamerv3-torch README).
"""
import subprocess
import sys
import os
import pathlib
import shutil

REPO_DIR = pathlib.Path(__file__).resolve().parent.parent / "dreamerv3-torch"
VENV_DIR = pathlib.Path(__file__).resolve().parent / "venv"

def venv_python():
    """Return path to venv python, OS-aware."""
    if os.name == "nt":
        return str(VENV_DIR / "Scripts" / "python.exe")
    return str(VENV_DIR / "bin" / "python")


def find_python311():
    """Find a Python 3.11 executable if available."""
    if sys.version_info.major == 3 and sys.version_info.minor == 11:
        return sys.executable
    
    candidates = [
        r"C:\Users\MSI\miniconda3\envs\kaggle_env\python.exe",
        "python3.11",
        "py -3.11",
    ]
    for c in candidates:
        try:
            res = subprocess.run([c, "--version"], capture_output=True, text=True)
            if res.returncode == 0 and "3.11" in res.stdout:
                return c
        except Exception:
            pass
    return sys.executable


def install_dependencies(py):
    req_file = REPO_DIR / "requirements.txt"
    if not req_file.exists():
        sys.exit(f"[ERROR] requirements.txt not found at {req_file}")

    print("1. Upgrading pip inside virtual environment...")
    subprocess.check_call([py, "-m", "pip", "install", "--upgrade", "pip"])

    print("2. Preparing dependencies (ensuring prebuilt binary wheels for Python 3.11 Windows)...")
    # Read original requirements.txt
    with open(req_file, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    
    # On Windows / Python 3.11, matplotlib 3.5.0 lacks cp311 wheels (wheels start at 3.6.0+).
    # Using matplotlib>=3.6.0 ensures instant binary wheel installation without requiring MSVC C++ tools.
    adapted_reqs = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("matplotlib==") and os.name == "nt" and sys.version_info >= (3, 11):
            adapted_reqs.append("matplotlib>=3.6.0")
        elif line.startswith("protobuf==") and os.name == "nt" and sys.version_info >= (3, 11):
            adapted_reqs.append("protobuf>=3.20.0")
        else:
            adapted_reqs.append(line)

    temp_req = pathlib.Path(__file__).resolve().parent / "requirements_installed.txt"
    with open(temp_req, "w", encoding="utf-8") as f:
        f.write("\n".join(adapted_reqs) + "\n")

    print("3. Installing packages from dreamerv3-torch requirements...")
    subprocess.check_call([py, "-m", "pip", "install", "--prefer-binary", "-r", str(temp_req)])

    print("4. Installing helper packages for test harness (psutil, ruamel.yaml, jupyter, nbconvert)...")
    subprocess.check_call([py, "-m", "pip", "install", "--prefer-binary", "psutil", "ruamel.yaml", "jupyter", "nbconvert", "ipykernel"])


def main():
    print("=" * 70)
    print("00: Environment Setup & Dependency Installation")
    print("=" * 70)

    # ── 1. Create or reset venv ───────────────────────────────────────────
    py = venv_python()
    if not pathlib.Path(py).exists():
        base_py = find_python311()
        print(f"Creating fresh virtual environment using {base_py} ...")
        if VENV_DIR.exists():
            shutil.rmtree(VENV_DIR, ignore_errors=True)
        subprocess.check_call([base_py, "-m", "venv", str(VENV_DIR)])

    py = venv_python()

    # ── 2. Install deps ───────────────────────────────────────────────────
    install_dependencies(py)

    # ── 3. Write versions.txt ─────────────────────────────────────────────
    version_script = r"""
import sys, os
lines = []
lines.append(f"python:      {sys.version}")
try:
    import torch
    lines.append(f"torch:       {torch.__version__}")
    lines.append(f"cuda_avail:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        lines.append(f"cuda_version:{torch.version.cuda}")
        lines.append(f"gpu_name:    {torch.cuda.get_device_name(0)}")
    else:
        lines.append("cuda_version:N/A (no GPU)")
        lines.append("gpu_name:    N/A")
except ImportError:
    lines.append("torch: NOT INSTALLED")
try:
    import dm_control
    lines.append(f"dm_control:  {dm_control.__version__}")
except Exception as e:
    lines.append(f"dm_control:  IMPORT ERROR: {e}")
try:
    import mujoco
    lines.append(f"mujoco:      {mujoco.__version__}")
except Exception as e:
    lines.append(f"mujoco:      IMPORT ERROR: {e}")

with open("versions.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\n--- Detected Environment Versions ---")
print("\n".join(lines))
"""
    subprocess.check_call([py, "-c", version_script],
                          cwd=str(pathlib.Path(__file__).resolve().parent))
    print("\n[OK] Setup complete. Versions written to versions.txt")


if __name__ == "__main__":
    main()
