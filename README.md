# Feasibility Study: MBRL Walker-Walk (15-Day Beginner Plan)

This repository contains the complete empirical feasibility study, test harness, evaluation artifacts, and patched reference codebase for the **World Walker Challenge** (training DreamerV3 on DeepMind Control Suite `walker_walk` with proprioceptive inputs under a 50,000-frame budget).

---

## Repository Structure

```
├── README.md                           # Main repository overview
├── .gitignore                          # Excludes large binaries (>100MB checkpoints) and venvs
├── dreamerv3-torch/                    # Reference MBRL codebase (commit 6ef8646d) with bug & budget fixes
│   ├── dreamer.py                      # Main training script (patched with strict budget enforcement)
│   ├── models.py                       # World model, actor-critic imagination loop
│   ├── networks.py                     # RSSM dynamics, continuous Gaussian vs discrete categorical latents
│   ├── configs.yaml                    # Environment, model, and training hyperparameters
│   └── requirements.txt                # Upstream requirements
└── feasibility_tests/                  # Self-contained feasibility evaluation harness
    ├── REPORT_v2.md                    # Comprehensive Feasibility Report v2 (Claims C1-C10, Criteria F1-F9)
    ├── README.md                       # Harness usage documentation
    ├── thresholds.json                 # Adjustable acceptance thresholds for F1-F9
    ├── versions.txt                    # Verified Python, PyTorch, CUDA, and MuJoCo versions
    ├── 00_setup.py / 00_setup.sh       # Automated venv creation & dependency setup
    ├── 01_baseline_run.py              # Baseline 50k run with resource monitoring
    ├── 02_step_counter_check.py        # Step counter semantics verification
    ├── 03_multiseed_baseline.py        # Multi-seed baseline evaluation
    ├── 04_variant_feasibility.py       # World model variant feasibility analysis
    ├── 05_open_loop_eval.py            # Initial open-loop prediction fidelity metric
    ├── 06_compute_budget.py            # Compute budget analysis
    ├── 07_fresh_install_test.py / .sh  # Clean environment installation test
    ├── 08_ps_consistency_check.py     # Problem statement consistency check
    ├── 09_human_pilot_checklist.md     # Days 1-3 beginner onboarding checklist
    ├── 10_make_report.py               # Automated report generator
    └── results_v2/                     # Upgraded Phase 2 Empirical Results & Artifacts
        ├── audit.md                    # Detailed audit of 9 discrepancies in v1
        ├── power_analysis.md           # Statistical power & minimum detectable difference analysis
        ├── budget_fix.patch            # Minimal patch file enforcing exact budget limits
        ├── open_loop_eval_v2.py        # Upgraded open-loop eval (Horizon 0 + persistence baseline)
        ├── open_loop_summary_v2.json   # Multi-seed open-loop fidelity results across horizons 0, 5, 15, 50
        ├── compute_budget_v2.py        # Compute budget analysis with hardware scaling factors
        ├── compute_budget_rtx4060.json # Budget projection on RTX 4060 GPU
        ├── compute_budget_t4_estimate.json # Budget projection on Nvidia T4 (Colab/Kaggle)
        ├── ps_consistency_check_v2.py  # Semantic mapping from PS text to C-claims & evidence
        ├── ps_consistency_report_v2.json
        ├── run_sequential_experiments.py # Resumable sequential GPU long-run orchestrator
        ├── demo_notebook.ipynb         # Student demo notebook (setup, smoke test, visualization)
        ├── timing_notebook.ipynb       # Colab/Kaggle benchmarking notebook
        ├── requirements_pinned.txt     # Pinned dependencies for clean Linux installation
        ├── install.sh                  # Linux headless rendering & dependency installer
        └── cleaned/                    # Cleaned and deduplicated baseline datasets & manifest
```

---

## Key Feasibility Findings

1. **Step Budget Semantics (C1, F1)**:
   In `dreamerv3-torch` with `dmc_proprio`, `action_repeat = 2`. The 50,000-step budget represents **50,000 simulator frames and 25,000 agent decisions**. A loop clamp patch ([results_v2/budget_fix.patch](feasibility_tests/results_v2/budget_fix.patch)) ensures training halts at exactly 50,000 frames with a final evaluation.

2. **World Model Variants (C3, C4)**:
   - **Continuous Gaussian vs Discrete Categorical Latents**: Fully supported natively via `--dyn_discrete 0` (0 code changes). Verified to run with finite KL divergence and zero NaNs.
   - **RSSM vs MLP Ensemble Dynamics**: Infeasible for beginners within 15 days (~150–250 lines across core files). Correctly removed from the problem statement.
   - **Optional Ablations**: Latent capacity (`--dyn_deter 256 --dyn_stoch 16`) and KL balancing scales (`--dyn_scale 1.0 --rep_scale 0.5`) verified to start and train cleanly.

3. **Statistical Power & Sample Size (C6, F7)**:
   DreamerV3 exhibits a high seed standard deviation ($s \approx 139\text{--}187$) on `walker_walk` at 50,000 frames. At $n = 3$ seeds, the Minimum Detectable Difference is **421 to 568 score points**. Beginners **cannot make claims of statistical superiority** at $n = 3$; results must be reported as exploratory distributions.

4. **Compute Budget & GPU Allocation (C5, F4)**:
   - **Core Plan** (1 Baseline + 1 Variant × 3 seeds = 6 runs): Requires **18.2 GPU-hours** on an RTX 4060 laptop (or ~24.6h on an Nvidia T4), fitting within weekly free GPU quotas (30h/week on Kaggle).
   - **Full Plan** (18 runs): Requires **55 to 74 GPU-hours**, which exceeds weekly quotas and must be pruned.

5. **Open-Loop Prediction Fidelity (C7, F6)**:
   Evaluated across Horizons 0, 5, 15, and 50. Horizon 0 (static posterior reconstruction) achieves an NMSE of 0.08 (92% variance explained). For Horizons 5–50, the world model consistently outperforms naive persistence baselines by 4x to 13x.

For the full detailed analysis and evidence citations, see [`feasibility_tests/REPORT_v2.md`](feasibility_tests/REPORT_v2.md).
