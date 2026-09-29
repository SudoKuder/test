# Feasibility Test Harness: MBRL Walker-Walk (15-Day Beginner Plan)

This self-contained test harness verifies whether the proposed Model-Based RL problem statement on DeepMind Control Suite `walker_walk` (proprioceptive inputs, 50k steps budget, 15-day timeline) is feasible for beginners.

- **Primary Codebase**: [`NM512/dreamerv3-torch`](https://github.com/NM512/dreamerv3-torch)
- **Verified Commit Hash**: `6ef8646d807cd10ce0c88e10a7e943211e7fc44c`
- **Supported Benchmark**: DeepMind Control Proprioceptive (`dmc_proprio`) with continuous actions and low-dimensional states.

---

## Test Execution Order & GPU Requirements Summary

| Script / Test | Description | Requires GPU? | Expected Runtime | Output Artifact |
| :--- | :--- | :---: | :---: | :--- |
| `00_setup.py` (or `00_setup.sh`) | Creates venv, installs pinned deps | **No** (CPU only) | ~3–5 min | `versions.txt` |
| `02_step_counter_check.py` | Deduces step counter semantics | **No** (Code parsing) | < 5 sec | `step_counter_report.txt` |
| `04_variant_feasibility.py` | Analyzes code for planned variants | **No** (Code analysis) | < 5 sec | `variant_feasibility.md` |
| `08_ps_consistency_check.py` | Checks numbers and text consistency | **No** (Text parsing) | < 5 sec | `ps_consistency_report.json` |
| `01_baseline_run.py` | Unmodified 50k baseline run (Seed 0) | **YES** (CUDA GPU) | ~30–40 min | `baseline_results.json` |
| `03_multiseed_baseline.py` | Multi-seed run (Seeds 0, 1, 2) | **YES** (CUDA GPU) | ~90–120 min | `multiseed_results.json` |
| `05_open_loop_eval.py` | Open-loop prediction fidelity metric | **YES** (or CPU eval) | ~10 sec | `open_loop_eval_results.json` |
| `06_compute_budget.py` | Computes required vs available GPU hours | **No** (Math/Analysis)| < 5 sec | `compute_budget_report.json` |
| `07_fresh_install_test.py` | Clean env smoke test + nbconvert | **YES** (GPU for smoke)| ~3–5 min | `fresh_install_result.txt` |
| `10_make_report.py` | Reads all artifacts & builds `REPORT.md`| **No** (Report generator)| < 5 sec | `REPORT.md` |

---

## Step-by-Step Execution Guide

Run all commands from within the `feasibility_tests/` folder:

```bash
cd feasibility_tests
```

### Phase 1: Environment Setup & Static Feasibility (No GPU Required)

1. **Setup virtual environment and verify dependencies**:
   ```bash
   python 00_setup.py
   ```
   *(On Linux/macOS/WSL, you can alternatively run `./00_setup.sh`)*

2. **Verify step counter semantics**:
   ```bash
   python 02_step_counter_check.py
   ```

3. **Verify world model variant feasibility (Gaussian latents vs MLP Ensemble)**:
   ```bash
   python 04_variant_feasibility.py
   ```

4. **Verify problem statement numerical consistency**:
   ```bash
   python 08_ps_consistency_check.py
   ```

---

### Phase 2: Dynamic Execution & Training (GPU Required)

5. **Run baseline single-seed test (50,000 steps)**:
   ```bash
   # Quick smoke test first (optional):
   python 01_baseline_run.py --steps 2000

   # Full 50k steps baseline run:
   python 01_baseline_run.py --steps 50000 --seed 0
   ```

6. **Run multi-seed baseline (3 seeds, 10 eval episodes per seed)**:
   ```bash
   python 03_multiseed_baseline.py --seeds 3 --steps 50000
   ```

7. **Run open-loop prediction fidelity evaluation (horizons 5, 15, 50)**:
   ```bash
   python 05_open_loop_eval.py --logdir logdir/walker_walk_seed0
   ```

---

### Phase 3: Budget Analysis, Clean Install Test & Final Report

8. **Evaluate compute budget & GPU quotas**:
   ```bash
   # Check against available GPU hours (e.g. 50 hours):
   python 06_compute_budget.py --available_gpu_hours 50.0 --num_variants 3 --num_ablations 2 --num_seeds 3
   ```

9. **Test fresh installation and notebook execution**:
   ```bash
   python 07_fresh_install_test.py
   # Or on bash: bash 07_fresh_install_test.sh
   ```

10. **Generate the comprehensive feasibility report**:
    ```bash
    python 10_make_report.py
    ```
    This compiles all results into `REPORT.md`.

---

## Artifacts & References
- [`09_human_pilot_checklist.md`](file:///c:/codes/reserch-a-thon%20test/feasibility_tests/09_human_pilot_checklist.md): Human pilot volunteer milestone checklist.
- [`REPORT.md`](file:///c:/codes/reserch-a-thon%20test/feasibility_tests/REPORT.md): Final synthesized feasibility report.
