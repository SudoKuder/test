#!/usr/bin/env bash
# 00_setup.sh - Create fresh virtualenv, install pinned dependencies, and write versions.txt
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/../dreamerv3-torch"
VENV_DIR="$SCRIPT_DIR/venv"

echo "=== Setting up Virtual Environment ==="
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR" || python -m venv "$VENV_DIR"
fi

# Activate venv
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
elif [ -f "$VENV_DIR/Scripts/activate" ]; then
    source "$VENV_DIR/Scripts/activate"
fi

echo "Upgrading pip and installing pinned dependencies..."
pip install --upgrade pip
pip install -r "$REPO_DIR/requirements.txt"
pip install psutil

echo "=== Writing versions.txt ==="
python -c "
import sys
lines = []
lines.append(f'python:      {sys.version}')
try:
    import torch
    lines.append(f'torch:       {torch.__version__}')
    lines.append(f'cuda_avail:  {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        lines.append(f'cuda_version:{torch.version.cuda}')
        lines.append(f'gpu_name:    {torch.cuda.get_device_name(0)}')
    else:
        lines.append('cuda_version:N/A (no GPU)')
        lines.append('gpu_name:    N/A')
except ImportError:
    lines.append('torch: NOT INSTALLED')
try:
    import dm_control
    lines.append(f'dm_control:  {dm_control.__version__}')
except Exception as e:
    lines.append(f'dm_control:  IMPORT ERROR: {e}')
try:
    import mujoco
    lines.append(f'mujoco:      {mujoco.__version__}')
except Exception as e:
    lines.append(f'mujoco:      IMPORT ERROR: {e}')

with open('versions.txt', 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('\n'.join(lines))
"

echo "Setup complete! Versions written to versions.txt"
