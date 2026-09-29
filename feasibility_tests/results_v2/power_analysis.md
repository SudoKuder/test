# Statistical Power & Minimum Detectable Difference (MDD) Analysis

**Date**: 2026-09-29  
**Subject**: Statistical feasibility of detecting differences between world-model configurations under the PS v2 design ($n=3$ seeds vs $n=5$ seeds, 50k frames).  
**Methodology**: Two-sample independent $t$-test power approximation ($\alpha = 0.05$, two-tailed, power $1 - \beta = 0.80$, degrees of freedom $\text{df} = 2n - 2$).

---

## 1. Measured Seed-Level Variance from Baseline Runs

From the three executed baseline seeds on `walker_walk` (dmc_proprio, action repeat 2):

### A. Evaluated Returns at End of Run (~55k–56k frames, 10 eval episodes per seed)
* **Seed 0 Mean**: 595.41
* **Seed 1 Mean**: 221.41
* **Seed 2 Mean**: 421.58
* **Mean of Seed Means**: **412.80**
* **Seed-to-Seed Sample Standard Deviation ($s$)**: **187.15**
* **Standard Error of the Mean ($SE = s/\sqrt{3}$)**: **108.05**
* **Bootstrap 95% Confidence Interval (100,000 resamples over seeds)**: **[221.41, 595.41]**

### B. Evaluated Returns at Last Checkpoint $\le 50,000$ Frames (Step 45,000–46,000)
* **Seed 0 (Step 46,000)**: 491.10
* **Seed 1 (Step 45,000)**: 222.18
* **Seed 2 (Step 45,000)**: 415.90
* **Mean**: **376.39**
* **Seed-to-Seed Sample Standard Deviation ($s$)**: **138.74**
* **Standard Error of the Mean ($SE = s/\sqrt{3}$)**: **80.10**
* **Bootstrap 95% Confidence Interval (100,000 resamples over seeds)**: **[222.18, 491.10]**

---

## 2. Minimum Detectable Difference (MDD) Calculations

The Minimum Detectable Difference ($\Delta$) represents the smallest true difference between two condition means (e.g., Baseline vs Variant) that a test will reliably detect ($80\%$ of the time at $\alpha = 0.05$):
$$\Delta = (t_{\alpha/2, \, \text{df}} + t_{\beta, \, \text{df}}) \cdot s \cdot \sqrt{\frac{2}{n}}$$
where:
* $\alpha = 0.05 \implies t_{\alpha/2, \, \text{df}}$ is the critical value for a two-tailed test ($97.5$th percentile).
* $1 - \beta = 0.80 \implies t_{\beta, \, \text{df}}$ is the critical value for $80\%$ power ($80$th percentile).
* $\text{df} = 2n - 2$.

### Summary Table

| Evaluation Basis | Seed SD ($s$) | Sample Size ($n$) | Degrees of Freedom ($\text{df}$) | Critical $t_{\alpha/2}$ | Critical $t_{\beta}$ | **Minimum Detectable Difference (MDD)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **End-of-Run Eval (10 eps)** | 187.15 | **$n = 3$** | 4 | 2.776 | 0.941 | **568.06 score points** |
| **End-of-Run Eval (10 eps)** | 187.15 | **$n = 5$** | 8 | 2.306 | 0.889 | **378.17 score points** |
| **Checkpoint $\le 50\text{k}$** | 138.74 | **$n = 3$** | 4 | 2.776 | 0.941 | **421.12 score points** |
| **Checkpoint $\le 50\text{k}$** | 138.74 | **$n = 5$** | 8 | 2.306 | 0.889 | **280.35 score points** |

---

## 3. Plain-Language Power Interpretation

> **Plain-Language Summary for Problem Statement Reviewers and Participants**:
> 
> In DeepMind Control `walker_walk`, episodic returns range from $0$ to $1,000$. Training DreamerV3 under a budget of only 50,000 frames yields high seed-to-seed variance: one seed may find an effective forward-locomotion gait early and achieve a score of $\sim 500\text{--}600$, while another seed gets trapped in a slow stumble or crouch and achieves only $\sim 220$ (seed standard deviation $s \approx 140\text{--}187$).
> 
> Because of this large variance, an experiment with only **$n = 3$ seeds per configuration has a Minimum Detectable Difference of 420 to 568 score points**. This means that a world-model variant would need to outperform the baseline by over **400 points** (e.g., scoring $800+$ compared to $376$) for the difference to be statistically distinguishable from random seed noise at $80\%$ power.
> 
> Typical real-world world-model algorithmic differences (such as discrete categorical vs continuous Gaussian representations) usually yield effect sizes between $30$ and $120$ score points. Therefore, **a 3-seed study has virtually zero statistical power to detect meaningful algorithmic differences**. Even expanding to $n = 5$ seeds only reduces the MDD to $\sim 280\text{--}380$ points.
> 
> **Explicit Rule for the PS**:
> Participants must be explicitly instructed **never to make claims of statistical significance or superiority** between configurations with $n=3$. Results must be presented as exploratory pilot distributions, reporting individual seed trajectories, per-seed means, and bootstrap confidence intervals that frankly communicate high uncertainty.
