# 09: Human Pilot Checklist (Days 1–3)

**Participant Profile**: 1 Volunteer with basic Python/ML knowledge (PyTorch tensors, training loops), zero prior Reinforcement Learning experience.  
**Goal**: Validate whether the setup, baseline execution, and initial evaluation metrics can be completed within the first 3 days of the 15-day timeline on Colab/Kaggle or local hardware.

---

## Participant Info
- **Pilot Volunteer Name**: __________________________
- **Start Date**: __________________________
- **Hardware / Platform**: [ ] Local GPU (Model: _________)  [ ] Google Colab  [ ] Kaggle Notebooks
- **Python Version**: __________________________
- **CUDA / Torch Version**: __________________________

---

## Day 1: Environment Setup & Smoke Test
*Target: Get the repository cloned, virtual environment configured, dependencies working, and verify a 1,000-step smoke run.*

- [ ] **Milestone 1.1**: Clone repository and verify commit hash (`6ef8646d807cd10ce0c88e10a7e943211e7fc44c`).
  - *Time Started*: `____-__-__ __:__` | *Time Finished*: `____-__-__ __:__`
  - *Status*: [ ] Pass [ ] Blocked
  - *Notes / Issues*:

- [ ] **Milestone 1.2**: Create virtual environment and install pinned dependencies (`requirements.txt`).
  - *Time Started*: `____-__-__ __:__` | *Time Finished*: `____-__-__ __:__`
  - *Status*: [ ] Pass [ ] Blocked
  - *Notes / Issues*:

- [ ] **Milestone 1.3**: Verify MuJoCo and DeepMind Control imports (`dm_control.suite.load('walker', 'walk')`).
  - *Time Started*: `____-__-__ __:__` | *Time Finished*: `____-__-__ __:__`
  - *Status*: [ ] Pass [ ] Blocked
  - *Notes / Issues*:

- [ ] **Milestone 1.4**: Run 1,000-step smoke test (`python 01_baseline_run.py --steps 1000`).
  - *Time Started*: `____-__-__ __:__` | *Time Finished*: `____-__-__ __:__`
  - *Status*: [ ] Pass [ ] Blocked
  - *Measured Smoke Run Time*: ________ seconds
  - *Notes / Issues*:

---

## Day 2: Baseline Run (50,000 Steps)
*Target: Understand configuration parameters (`dmc_proprio`, `action_repeat: 2`), launch full 50k baseline training run (Seed 0), and monitor resources.*

- [ ] **Milestone 2.1**: Understand step counter semantics (`02_step_counter_check.py`).
  - *Time Started*: `____-__-__ __:__` | *Time Finished*: `____-__-__ __:__`
  - *Status*: [ ] Pass [ ] Blocked
  - *Confirmed Understanding*: 50,000 env steps = 25,000 decisions = 50,000 simulator frames.

- [ ] **Milestone 2.2**: Launch full 50,000-step baseline run (`python 01_baseline_run.py --steps 50000 --seed 0`).
  - *Time Started*: `____-__-__ __:__` | *Time Finished*: `____-__-__ __:__`
  - *Status*: [ ] Pass [ ] Blocked
  - *Measured Wall-Clock Duration*: ________ minutes
  - *Peak GPU Memory*: ________ MB | *Peak CPU RAM*: ________ MB

- [ ] **Milestone 2.3**: Verify artifacts saved (`latest.pt`, `metrics.jsonl`, `train_eps/*.npz`, `eval_eps/*.npz`).
  - *Time Started*: `____-__-__ __:__` | *Time Finished*: `____-__-__ __:__`
  - *Status*: [ ] Pass [ ] Blocked
  - *Notes / Issues*:

---

## Day 3: Evaluation, Metrics, and Variant Planning
*Target: Parse evaluation logs, compute first learning curves, calculate pooled IQM/Mean, run open-loop prediction fidelity evaluation, and plan world-model modifications.*

- [ ] **Milestone 3.1**: Inspect `metrics.jsonl` and plot the evaluation return curve.
  - *Time Started*: `____-__-__ __:__` | *Time Finished*: `____-__-__ __:__`
  - *Final Baseline Return (Seed 0)*: ________
  - *Status*: [ ] Pass [ ] Blocked

- [ ] **Milestone 3.2**: Execute open-loop prediction fidelity metric (`python 05_open_loop_eval.py`).
  - *Time Started*: `____-__-__ __:__` | *Time Finished*: `____-__-__ __:__`
  - *State MSE*: H5: ______ | H15: ______ | H50: ______
  - *Reward MSE*: H5: ______ | H15: ______ | H50: ______
  - *Status*: [ ] Pass [ ] Blocked

- [ ] **Milestone 3.3**: Review World Model Variant Feasibility report (`variant_feasibility.md`).
  - *Time Started*: `____-__-__ __:__` | *Time Finished*: `____-__-__ __:__`
  - *Decision*: Adopt Continuous Latents (`--dyn_discrete 0`); defer or replace MLP Ensemble with parameter ablations.
  - *Status*: [ ] Pass [ ] Blocked

---

## Qualitative Feedback & Blocker Log

| Date / Milestone | Description of Blocker / Frustration | Resolution / Workaround | Time Lost |
| :--- | :--- | :--- | :--- |
| | | | |
| | | | |
| | | | |

### General Assessment
1. *Was the setup documentation clear and error-free?*  
   [ ] Yes  [ ] Mostly  [ ] Needs Improvement (Details: ______________________________________)
2. *Did training run within expected time limits without crashing?*  
   [ ] Yes  [ ] No (Details: ______________________________________)
3. *Did the participant feel confident proceeding to Day 4 (variants)?*  
   [ ] Yes  [ ] Hesitant  [ ] No
