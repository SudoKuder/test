# Comprehensive Audit of Feasibility Tests v1

**Date**: 2026-09-29  
**Auditor**: ML Research-Ops Engineering  
**Subject**: Discrepancies between raw experimental data / codebase implementation and the claims in `REPORT.md` (and related test harness v1 scripts).

---

## 1. Executive Summary of Audit

An exhaustive comparison was performed between the codebase (`NM512/dreamerv3-torch` commit `6ef8646d807cd10ce0c88e10a7e943211e7fc44c`), the test harness artifacts in `feasibility_tests/`, and the summary assertions in `REPORT.md`.

Nine critical discrepancies and hygiene defects were identified:
1. **Unenforced Budget**: `dreamer.py` trained beyond the 50,000-frame budget up to 55,000–56,000 frames; `REPORT.md` claimed it stopped at 25,000 decisions.
2. **Post-Budget Return Inflation & Hidden Seed Variance**: Returns reported as "final" were evaluated at 55k–56k frames, not at the $\le 50\text{k}$ budget limit. Furthermore, pooling 30 evaluation episodes masked a high seed-to-seed standard deviation ($\sigma \approx 187$).
3. **Overlapping GPU Execution**: Seed 0's wall-clock duration overlapped with a secondary run (`test_step50.json`), inflating runtime and peak VRAM to 7.8 GB.
4. **Target Platform Mismatch & Masked Errors**: The fresh installation test ran on Windows (not Linux Colab/Kaggle) and swallowed `pip` stderr on failure before falling back to relaxed constraints.
5. **Uninterpretable Prediction Fidelity**: The open-loop evaluation lacked a horizon-0 (reconstruction) baseline and a persistence baseline.
6. **Data Hygiene & Mislabeled Artifacts**: Evaluated checkpoints contained 4x duplicates; `test_step50.json` had `steps: 50` but 36,000 steps of data; `baseline_seed0.json` recorded an unhandled 1.5 s failure; `peak_cpu_mem_mb` was consistently 0.0.
7. **Unverified Variant**: `--dyn_discrete 0` (continuous Gaussian latents) was never run or empirically verified; its feasibility was asserted purely via code inspection.
8. **Skewed Compute Budget Projections**: Compute estimations relied on single-run or mislabeled timing artifacts rather than sequential multi-seed distribution medians.
9. **Superficial Consistency Validation**: `08_ps_consistency_check.py` performed regex text matching on `ps.txt` without checking algorithmic semantics.

---

## 2. Detailed Discrepancy Breakdown

### Discrepancy 1: Budget Enforcement Flaw
* **Code Reference**: `dreamer.py` line 306:
  ```python
  while agent._step < config.steps + config.eval_every:
  ```
* **Mechanism**: With `config.steps = 25000` decisions (50,000 simulator frames at `action_repeat = 2`) and `config.eval_every = 5000` decisions (10,000 frames), the condition allows training until `agent._step >= 30000` decisions (60,000 frames).
* **Observed Reality**:
  - Seed 0: Trained to step 56,000 (28,000 decisions).
  - Seed 1: Trained to step 55,000 (27,500 decisions).
  - Seed 2: Trained to step 55,000 (27,500 decisions).
* **REPORT.md Erroneous Claim** (Section 2, Line 28): *"dreamer.py (line 302): while agent._step < config.steps + config.eval_every (stops at 25,000 decisions)"*.
* **Verdict**: **FALSE**. The run exceeded the budget by 5,000 to 6,000 frames (10% to 12% over budget).

---

### Discrepancy 2: Post-Budget Return Inflation & Hidden Seed Variance
* **Observed Reality in `metrics.jsonl`**:
  - **Seed 0**:
    - Last checkpoint $\le 50\text{k}$ frames: **Step 46,000 $\rightarrow$ Eval Return = 491.10**
    - Post-budget checkpoint: Step 56,000 $\rightarrow$ Eval Return = 600.85
  - **Seed 1**:
    - Last checkpoint $\le 50\text{k}$ frames: **Step 45,000 $\rightarrow$ Eval Return = 222.18**
    - Post-budget checkpoint: Step 55,000 $\rightarrow$ Eval Return = 225.74
  - **Seed 2**:
    - Last checkpoint $\le 50\text{k}$ frames: **Step 45,000 $\rightarrow$ Eval Return = 415.90**
    - Post-budget checkpoint: Step 55,000 $\rightarrow$ Eval Return = 445.32
* **Seed-to-Seed Variance**:
  - The seed means of the final 10 evaluation episodes are:
    - Seed 0: $595.41 \pm 41.66$
    - Seed 1: $221.41 \pm 16.28$
    - Seed 2: $421.58 \pm 46.82$
  - Mean of seed means: **412.80**
  - **Seed-to-seed sample standard deviation**: **187.15** ($SE = 108.05$).
* **REPORT.md Erroneous Claim**:
  - `REPORT.md` Section 3 cited Final Return as `600.85` (which was achieved at 56k frames, well outside the 50k budget).
  - `multiseed_results.json` pooled all 30 episodes directly to report a pooled standard deviation of $157.32$ and an IQM of $410.94$, completely ignoring the hierarchical grouping where **seed** is the experimental unit of randomization.

---

### Discrepancy 3: Overlapping GPU Execution on Seed 0
* **Observed Reality**:
  - In `test_step50.json`, timestamp is `2026-09-29 04:46:50` with wall clock `10026.0 s` and `peak_gpu_mem_mb: 7837`.
  - In `baseline_results.json`, timestamp is `2026-09-29 06:07:36` with wall clock `11137.3 s` and `peak_gpu_mem_mb: 5284`.
  - Normal single-run peak GPU memory on this codebase is $\sim 4.7\text{--}5.1\text{ GB}$ (as confirmed by Seed 1 at 5123 MB and Seed 2 at 4731 MB). Peak VRAM reaching 7837 MB indicates two processes occupied the 8 GB RTX 4060 concurrently.
  - The runtime of Seed 0 (11,137 s) was likely inflated by compute/PCIe contention.
  - Sequential runs: Seed 1 = 10,927.8 s (3.04 h), Seed 2 = 8,974.2 s (2.49 h).

---

### Discrepancy 4: Fresh Install Platform Mismatch & Silenced Errors
* **Observed Reality**:
  - `07_fresh_install_test.py` executed on Windows 11 (AMD64) using the local Python 3.11 environment.
  - When `pip install -r requirements.txt` failed due to missing prebuilt wheels for `matplotlib==3.5.0` on Python 3.11 Windows, the script caught the exception but discarded `res.stderr`, printing only:
    `[INFO] Direct install failed on unbuilt wheels; applying binary wheel compatibility...`
  - Target student platform is **Linux x86_64** (Google Colab / Kaggle). Testing exclusively on Windows does not establish whether `pip install -r requirements.txt` succeeds cleanly on Ubuntu 22.04 with standard wheels.

---

### Discrepancy 5: Open-Loop Metric Lacked Baselines
* **Observed Reality**:
  - `05_open_loop_eval.py` measured state MSE and reward MSE at horizons 5, 15, and 50.
  - No measurement at horizon 0 (one-step reconstruction from latent posterior). Without horizon 0, it is impossible to separate world model dynamics drift from static decoder reconstruction error.
  - No persistence baseline (predicting that the state remains at $s_{\text{warmup}-1}$ and reward remains at $r_{\text{warmup}-1}$). Without a persistence baseline, an MSE of 13.96 cannot be contextualized (it could be worse than predicting a constant).

---

### Discrepancy 6: Harness Hygiene & Mislabeled Files
1. **Duplicate Checkpoints**: In `baseline_results.json`, step 5000 appears 4 times:
   `[{"step": 5000, "eval_return": 29.25}, {"step": 5000, "eval_return": 28.53}, {"step": 5000, "eval_return": 28.53}, {"step": 5000, "eval_return": 28.53}, ...]`
   Caused by naive parsing of `metrics.jsonl` from an aborted and restarted run.
2. **Mislabeled Run**: `test_step50.json` claims `"steps": 50`, but records eval checkpoints through step 36,000 and 10,026 seconds of runtime.
3. **Failed Run Artifact**: `baseline_seed0.json` recorded an unhandled 1.5-second crash (`success: false`, `final_return: null`) from before the OSMesa patch.
4. **Zeroed CPU Memory**: `peak_cpu_mem_mb: 0.0` in all summary JSON files. The polling thread failed to import `psutil` or resolve active child PIDs.
5. **False Positive on Pilot**: Test 09 was marked **PASS** in `REPORT.md` simply because `09_human_pilot_checklist.md` existed on disk, despite zero pilot trials being conducted.

---

### Discrepancy 7: Continuous Latents Variant Was Never Run
* In v1, `--dyn_discrete 0` was identified in `configs.yaml` line 36 and `networks.py` lines 161–172 as a config-level toggle.
* However, no training or smoke test was ever launched with `--dyn_discrete 0`. Whether it runs without numerical instabilities, NaNs in continuous Gaussian KL divergence, or tensor shape mismatches was **completely unverified**.

---

### Discrepancy 8: Compute Budget Projections
* `06_compute_budget.py` initially read `10026.0 s` from the mislabeled `test_step50.json` file.
* It ignored the distribution of sequential runs (Seed 1 = 10,928 s; Seed 2 = 8,974 s).
* Projections must use the median and range across all verified sequential runs, and must provide a hardware translation factor for target platforms (e.g. Nvidia T4/P100 vs RTX 4060).

---

### Discrepancy 9: Problem Statement Consistency Validation
* `08_ps_consistency_check.py` evaluated `ps.txt` using regex pattern matching on isolated words (e.g., checking if the substring "50,000" and "walker_walk" appeared).
* It did not verify whether the codebase's step counter matched the semantic definition in the PS text.

---

## 3. Corrected Inventory Manifest

The corrupted/intermediate artifacts have been preserved in their original state and cleaned copies created in `results_v2/cleaned/`:

| Original File | Status / Issue | Action Taken in `results_v2/cleaned/` |
| :--- | :--- | :--- |
| `baseline_results.json` | 4x duplicate at step 5000; includes post-budget step 56k | Deduplicated; created `baseline_results_cleaned.json` with separate $\le 50\text{k}$ and final entries. |
| `test_step50.json` | Mislabeled `steps: 50`; partial run (up to 36k) | Excluded from baseline analysis; archived with label `mislabeled_partial_run`. |
| `baseline_seed0.json` | Failed 1.5s OSMesa crash | Excluded from baseline statistics; noted as crash artifact. |
| `baseline_seed1.json` | Valid sequential run; trained to 55k | Retained; noted $\le 50\text{k}$ checkpoint = 222.18 (step 45k), final = 225.74 (step 55k). |
| `baseline_seed2.json` | Valid sequential run; trained to 55k | Retained; noted $\le 50\text{k}$ checkpoint = 415.90 (step 45k), final = 445.32 (step 55k). |
| `multiseed_results.json` | Pooled across 30 episodes, hiding seed variance | Recomputed hierarchical seed-level statistics in `multiseed_results_cleaned.json`. |
