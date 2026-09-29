#!/usr/bin/env python3
"""
05_open_loop_eval.py

Prediction Fidelity Metric:
Evaluates open-loop prediction accuracy of a trained world model checkpoint.

Workflow:
1. Load real trajectory sequences from the replay buffer (train_eps / eval_eps).
2. Warmup: Feed initial observations (e.g. 5 steps) to infer initial posterior state s_0.
3. Open-loop Unroll: Roll the world model forward using only true actions (no future observations).
4. Compute Mean Squared Error (MSE) for:
   - Reconstructed proprioceptive state
   - Predicted reward
   at prediction horizons h = 5, 15, and 50 steps.
5. Verify whether the replay buffer and dataset configuration supply sequences >= 50 steps.
"""
import argparse
import json
import os
import pathlib
import sys

# Auto-redispatch if torch is not available in the current environment
try:
    import torch
    import torch.nn.functional as F
except ImportError:
    venv_py = pathlib.Path(__file__).resolve().parent / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if venv_py.exists():
        import subprocess
        sys.exit(subprocess.call([str(venv_py), __file__] + sys.argv[1:]))
    else:
        sys.exit("[ERROR] torch is not installed in the current environment, and feasibility_tests/venv was not found. Please run 'python 00_setup.py' first.")

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent / "dreamerv3-torch"
sys.path.append(str(REPO))

import tools
import models
from dreamer import make_dataset, make_env

def load_yaml_file(filepath):
    try:
        from ruamel.yaml import YAML
        yaml_parser = YAML(typ="safe", pure=True)
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml_parser.load(f)
    except Exception:
        import yaml
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value

def load_config():
    """Load configuration matching the dmc_proprio walker_walk setup."""
    all_configs = load_yaml_file(REPO / "configs.yaml")
    import copy
    cfg = copy.deepcopy(all_configs["defaults"])
    recursive_update(cfg, all_configs["dmc_proprio"])
    cfg["task"] = "dmc_walker_walk"
    if "size" in cfg:
        cfg["size"] = tuple(cfg["size"])
    
    class ConfigDict(dict):
        def __getattr__(self, name): return self.get(name)
        def __setattr__(self, name, value): self[name] = value
    return ConfigDict(cfg)

def main():
    parser = argparse.ArgumentParser(description="Open-Loop Prediction Fidelity Evaluation")
    parser.add_argument("--logdir", type=str, default="logdir/walker_walk_seed0",
                        help="Path to training logdir containing latest.pt and train_eps/")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup steps for initial state inference")
    parser.add_argument("--out", type=str, default="open_loop_eval_results.json", help="Output JSON results file")
    args = parser.parse_args()

    print("=" * 70)
    print("05: Open-Loop Prediction Fidelity Evaluation")
    print("=" * 70)

    logdir = HERE / args.logdir if not pathlib.Path(args.logdir).is_absolute() else pathlib.Path(args.logdir)
    ckpt_path = logdir / "latest.pt"

    config = load_config()
    
    # ── Check Replay Buffer Capacity for 50-step horizons ──────────────
    required_horizon = 50
    total_needed_seq = args.warmup + required_horizon
    print(f"\n[Replay Capacity Verification]")
    print(f"- Dataset batch_length: {config.batch_length} steps")
    print(f"- Minimum sequence required (warmup {args.warmup} + horizon {required_horizon}): {total_needed_seq} steps")
    
    if config.batch_length >= total_needed_seq:
        print(f"[OK] Config batch_length ({config.batch_length}) is sufficient for 50-step horizon.")
        capacity_ok = True
    else:
        print(f"[WARN] Config batch_length ({config.batch_length}) is LESS than required {total_needed_seq} steps.")
        capacity_ok = False

    if not ckpt_path.exists():
        print(f"\n[INFO] Checkpoint not found at {ckpt_path}.")
        print("To run evaluation, first train a baseline model using 01_baseline_run.py.")
        dummy_result = {
            "status": "SKIPPED_NO_CHECKPOINT",
            "capacity_verified": capacity_ok,
            "batch_length": config.batch_length,
            "message": f"Checkpoint {ckpt_path} not found. Run baseline training first."
        }
        with open(HERE / args.out, "w", encoding="utf-8") as f:
            json.dump(dummy_result, f, indent=2)
        return

    # ── Setup Model and Device ─────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device
    print(f"Evaluating on device: {device}")

    # Build dummy environment to retrieve observation & action specs
    env = make_env(config, "eval", 0)
    acts = env.action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    world_model = models.WorldModel(env.observation_space, env.action_space, 0, config).to(device)

    # Load trained weights
    ckpt = torch.load(ckpt_path, map_location=device)
    wm_state_dict = {}
    for k, v in ckpt["agent_state_dict"].items():
        if k.startswith("_wm."):
            wm_state_dict[k[4:]] = v
    world_model.load_state_dict(wm_state_dict, strict=False)
    world_model.eval()

    # ── Load Replay Sequences ──────────────────────────────────────────
    eps_dir = logdir / "train_eps"
    if not eps_dir.exists() or not list(eps_dir.glob("*.npz")):
        eps_dir = logdir / "eval_eps"
    
    episodes = tools.load_episodes(eps_dir, limit=5)
    if not episodes:
        print(f"No episode files found in {eps_dir}")
        return

    dataset = make_dataset(episodes, config)
    batch = next(dataset)
    batch = world_model.preprocess(batch)

    # ── Open-Loop Rollout ──────────────────────────────────────────────
    warmup = args.warmup
    with torch.no_grad():
        # Step 1: Infer initial posterior state from warmup steps
        embed = world_model.encoder(batch)
        states, _ = world_model.dynamics.observe(
            embed[:, :warmup], batch["action"][:, :warmup], batch["is_first"][:, :warmup]
        )
        init_state = {k: v[:, -1] for k, v in states.items()}

        # Step 2: Roll forward open-loop using only actions
        actions = batch["action"][:, warmup:]
        prior = world_model.dynamics.imagine_with_action(actions, init_state)
        feat = world_model.dynamics.get_feat(prior)

        # Step 3: Decode predictions
        pred_reward = world_model.heads["reward"](feat).mode()
        true_reward = batch["reward"][:, warmup:]
        pred_obs = world_model.heads["decoder"](feat)

    # ── Compute Horizon Errors ─────────────────────────────────────────
    horizons = [5, 15, 50]
    metrics = {}
    print("\n--- Open-Loop Prediction Fidelity (MSE) ---")
    print(f"{'Horizon (steps)':<18} | {'State MSE':<15} | {'Reward MSE':<15}")
    print("-" * 55)

    max_len = prior["deter"].shape[1]
    for h in horizons:
        if h > max_len:
            print(f"Horizon {h:<10}: Sequence shorter than horizon ({max_len} steps).")
            continue
        h_idx = h - 1

        # Reward MSE at horizon h
        rew_mse = float(F.mse_loss(pred_reward[:, h_idx].squeeze(), true_reward[:, h_idx].squeeze()).cpu().item())

        # State MSE across decoded observation keys
        state_mse = 0.0
        keys_count = 0
        for k, dist in pred_obs.items():
            if k in batch:
                pred_s = dist.mode()
                true_s = batch[k][:, warmup:]
                state_mse += float(F.mse_loss(pred_s[:, h_idx], true_s[:, h_idx]).cpu().item())
                keys_count += 1
        state_mse /= max(1, keys_count)

        metrics[f"horizon_{h}"] = {
            "steps": h,
            "state_mse": round(state_mse, 6),
            "reward_mse": round(rew_mse, 6)
        }
        print(f"{h:<18} | {state_mse:<15.6f} | {rew_mse:<15.6f}")

    results = {
        "status": "COMPLETED",
        "device": str(device),
        "capacity_verified": capacity_ok,
        "warmup_steps": warmup,
        "metrics": metrics
    }
    
    out_file = HERE / args.out
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved open-loop evaluation results to {out_file}")

if __name__ == "__main__":
    main()
