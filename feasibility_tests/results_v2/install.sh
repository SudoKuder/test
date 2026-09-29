#!/usr/bin/env bash
# ==============================================================================
# install.sh - Clean Linux (Colab / Kaggle / Ubuntu) Installation Script
# ==============================================================================
set -euo pipefail

echo "=== 1. Checking Python Environment ==="
python3 --version

echo "=== 2. Upgrading Pip ==="
pip install --upgrade pip

echo "=== 3. Installing System Dependencies for Headless Rendering ==="
if command -v apt-get &> /dev/null; then
    echo "Detected apt package manager. Installing OSMesa and OpenGL utilities..."
    sudo apt-get update -qq || true
    sudo apt-get install -y -qq libosmesa6-dev libgl1-mesa-glx libglfw3 libglew-dev || true
fi

echo "=== 4. Installing Pinned Python Requirements ==="
pip install -r requirements_pinned.txt

echo "=== 5. Verifying Python Imports & Hardware Acceleration ==="
python3 -c "
import torch
print('PyTorch Version:', torch.__version__)
print('CUDA Available: ', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU Device:    ', torch.cuda.get_device_name(0))

import mujoco
print('MuJoCo Version: ', mujoco.__version__)

import dm_control
print('dm_control:     Loaded successfully')

import OpenGL
print('PyOpenGL:       Loaded successfully')
"

echo "=== Setup Complete. Ready for training & evaluation. ==="
