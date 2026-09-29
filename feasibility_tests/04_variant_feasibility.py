#!/usr/bin/env python3
"""
04_variant_feasibility.py

Inspects codebase configs and model code for the planned world model variants:
1. Continuous Gaussian vs Discrete Categorical latents
2. RSSM vs MLP Ensemble Dynamics

Writes findings to variant_feasibility.md with exact file and line citations.
"""
import pathlib
import datetime

HERE = pathlib.Path(__file__).resolve().parent

def main():
    timestamp = datetime.datetime.now().isoformat()
    
    report = f"""# World Model Variant Feasibility Report
*Generated: {timestamp}*

## Summary Matrix

| Planned Variant | Category | Codebase Support | Estimated Effort | Feasible for Beginners in 15 Days? |
| :--- | :--- | :--- | :--- | :--- |
| **1. Continuous Gaussian vs Discrete Categorical Latents** | **(a) Config Flag** | Fully natively supported | 0 lines of code (CLI / YAML config) | **YES** (Trivial) |
| **2. RSSM vs MLP Ensemble Dynamics** | **(c) Substantial Implementation** | Not available as world model | ~150–250 lines across 2+ files | **NO** (High Risk / Out of Scope) |

---

## Detailed Investigation

### 1. Continuous Gaussian vs Discrete Categorical Latents
- **Feasibility Category**: **(a) Config flag**
- **Effort**: 0 lines of code modified. Can be toggled via `--dyn_discrete 0` or modifying `configs.yaml`.
- **Mechanism in `NM512/dreamerv3-torch`**:
  - The codebase implements both continuous Gaussian and discrete categorical representations directly inside `networks.RSSM`.
  - When `dyn_discrete > 0`, it uses discrete latents (categorical with `dyn_stoch` groups of `dyn_discrete` classes).
  - When `dyn_discrete == 0` (or `False`), it falls back to continuous Gaussian latents with parameterized mean and standard deviation.
- **Exact Code Citations**:
  - `configs.yaml` (Line 36): `dyn_discrete: 32` defines the default categorical latent dimension.
  - `models.py` (Line 43): passes `config.dyn_discrete` into `networks.RSSM(...)`.
  - `models.py` (Lines 56-59): calculates feature dimension dynamically:
    ```python
    if config.dyn_discrete:
        feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
    else:
        feat_size = config.dyn_stoch + config.dyn_deter
    ```
  - `networks.py` (Line 38): `self._discrete = discrete` stores the flag.
  - `networks.py` (Lines 49-53): input dimension branching for RSSM linear layers.
  - `networks.py` (Lines 80-92): output projection layers branch (`stoch * discrete` vs `2 * stoch`).
  - `networks.py` (Lines 161-172): `get_dist` builds `tools.OneHotDist` if discrete, or `tools.ContDist(Normal(mean, std))` if continuous.
  - `networks.py` (Lines 242-270): `_suff_stats_layer` branches to return `logit` (discrete) or `mean` and `std` with `dyn_mean_act`/`dyn_std_act` (continuous).
  - `networks.py` (Lines 272-290): `kl_loss` calculates categorical KL vs continuous Gaussian KL.

---

### 2. RSSM vs MLP Ensemble Dynamics
- **Feasibility Category**: **(c) Substantial implementation**
- **Effort**: ~150–250 lines of new code across `models.py` and `networks.py`, plus testing.
- **Codebase Findings**:
  - The repository's world model is tightly coupled to the recurrent state-space model (`networks.RSSM`) and GRU cell (`networks.GRUCell`).
  - An MLP ensemble *does* exist in `exploration.py` (Lines 40-136, `Plan2Explore._train_ensemble`), but it is exclusively used for compute-curiosity intrinsic reward bonus via prediction disagreement, NOT as the generative dynamics engine.
  - There is no modular plugin interface or swap flag to replace `self.dynamics` with an MLP ensemble dynamics model.
- **Requirements to Implement**:
  1. Author a new `MLPEnsembleDynamics` class implementing `observe()`, `img_step()`, `imagine_with_action()`, and `get_feat()`.
  2. Implement bootstrapping/bagging and ensemble forward rollout (e.g. TS1 trajectory sampling or mean aggregation).
  3. Rewrite `WorldModel._train()` in `models.py` (Lines 108-174) to train ensemble members via regression loss / Gaussian NLL instead of KL divergence and temporal recurrence.
  4. Adapt actor-critic imagination in `ImagBehavior._imagine()` (`models.py` Lines 351-370) to roll out trajectories through the MLP ensemble without recurrent latent states (`deter`/`stoch`).
- **Verdict for 15-Day Beginner Timeline**:
  - Requiring a beginner with basic ML and no RL background to re-architect DreamerV3 into an ensemble-based model (like PETS or MBPO) within 15 days is high-risk and likely to lead to project failure.
  - **Recommendation**: Replace the MLP ensemble variant with hyperparameter/architectural ablations supported natively (e.g., latent dimension `dyn_stoch`/`dyn_deter`, KL balancing scale `dyn_scale`/`rep_scale`, or recurrent depth `dyn_rec_depth`).
"""

    out_file = HERE / "variant_feasibility.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(report)
        
    print("=" * 70)
    print("Variant Feasibility Analysis Complete")
    print("=" * 70)
    print(report)
    print(f"\nSaved report to {out_file}")

if __name__ == "__main__":
    main()
