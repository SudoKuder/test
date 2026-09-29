#!/bin/bash
# 07_fresh_install_test.sh
# Tests fresh install, smoke test, and nbconvert.

echo "--- Running Fresh Install Test ---"
export TEST_DIR="fresh_test_env"
rm -rf $TEST_DIR
mkdir $TEST_DIR
cd $TEST_DIR

echo "Cloning..."
git clone https://github.com/NM512/dreamerv3-torch.git
cd dreamerv3-torch

echo "Creating venv..."
python -m venv test_venv
source test_venv/Scripts/activate || source test_venv/bin/activate

echo "Installing..."
pip install -r requirements.txt

echo "Running smoke test (1000 steps)..."
python dreamer.py --configs dmc_proprio --task dmc_walker_walk --logdir ./smoke_log --steps 1000 > smoke_out.txt 2> smoke_err.txt
SMOKE_CODE=$?

if [ $SMOKE_CODE -eq 0 ]; then
    echo "Smoke test PASSED."
else
    echo "Smoke test FAILED. Error log:"
    cat smoke_err.txt
fi

echo "Creating dummy notebook to test nbconvert..."
pip install jupyter nbconvert
cat << 'EOF' > dummy.ipynb
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "import torch\n",
    "import dm_control\n",
    "print('Notebook works')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

jupyter nbconvert --to notebook --execute dummy.ipynb > nb_out.txt 2> nb_err.txt
NB_CODE=$?

if [ $NB_CODE -eq 0 ]; then
    echo "Notebook test PASSED."
else
    echo "Notebook test FAILED. Error log:"
    cat nb_err.txt
fi

cd ../..
if [ $SMOKE_CODE -eq 0 ] && [ $NB_CODE -eq 0 ]; then
    echo "OVERALL: PASS" > fresh_install_result.txt
else
    echo "OVERALL: FAIL" > fresh_install_result.txt
fi
