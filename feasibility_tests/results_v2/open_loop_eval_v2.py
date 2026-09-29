#!/usr/bin/env python3
"""
open_loop_eval_v2.py  -  Upgraded Open-Loop Prediction Fidelity Evaluation

Measures open-loop prediction accuracy of a trained world model checkpoint:
1. Horizons: 0 (one-step reconstruction from latent posterior), 5, 15, 50.
2. Baselines: Persistence baseline (predict last observed state and copy last observed reward).
3. Normalization: Reports both raw MSE and Normalized MSE (NMSE = MSE / Var(ground_truth)).
4. Warmup: Fixed warmup steps (default 5 steps) with fixed evaluation seed.
5. Verification: Confirms replay capacity >= warmup + max(horizons).
"""
import argparse
import copy
import json
import os
import pathlib
import sys
import numpy as np

# Ensure torch is available
try:
    import torch
    import torch.nn.functional as F
except ImportError:
    venv_py = pathlib.Path(__file__).resolve().parent.parent / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if venv_py.exists():
        import subprocess
        sys.exit(subprocess.call([str(venv_py), __file__] + sys.argv[1:]))
    else:
        sys.exit("[ERROR] torch is not installed in the current environment.")

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent / "dreamerv3-torch"
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

def load_config(dyn_discrete=None):
    all_configs = load_yaml_file(REPO / "configs.yaml")
    cfg = copy.deepcopy(all_configs["defaults"])
    recursive_update(cfg, all_configs["dmc_proprio"])
    cfg["task"] = "dmc_walker_walk"
    if dyn_discrete is not None:
        cfg["dyn_discrete"] = int(dyn_discrete)
    if "size" in cfg:
        cfg["size"] = tuple(cfg["size"])
    
    class ConfigDict(dict):
        def __getattr__(self, name): return self.get(name)
        def __setattr__(self, name, value): self[name] = value
    return ConfigDict(cfg)

def evaluate_checkpoint(logdir, warmup=5, out_file=None, dyn_discrete=None):
    logdir = pathlib.Path(logdir).resolve()
    ckpt_path = logdir / "latest.pt"

    config = load_config(dyn_discrete)
    max_horizon = 50
    total_needed_seq = warmup + max_horizon
    capacity_ok = config.batch_length >= total_needed_seq

    if not ckpt_path.exists():
        res = {
            "status": "SKIPPED_NO_CHECKPOINT",
            "capacity_verified": capacity_ok,
            "message": f"Checkpoint not found at {ckpt_path}"
        }
        if out_file:
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
        return res

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device

    # Set seeds for reproducible evaluation sequence sampling
    torch.manual_seed(42)
    np.random.seed(42)

    env = make_env(config, "eval", 0)
    acts = env.action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    world_model = models.WorldModel(env.observation_space, env.action_space, 0, config).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    wm_state_dict = {}
    for k, v in ckpt["agent_state_dict"].items():
        if k.startswith("_wm."):
            wm_state_dict[k[4:]] = v
    world_model.load_state_dict(wm_state_dict, strict=False)
    world_model.eval()

    # Load episode data
    eps_dir = logdir / "train_eps"
    if not eps_dir.exists() or not list(eps_dir.glob("*.npz")):
        eps_dir = logdir / "eval_eps"
    
    episodes = tools.load_episodes(eps_dir, limit=10)
    if not episodes:
        raise RuntimeError(f"No episode files found in {eps_dir}")

    dataset = make_dataset(episodes, config)
    batch = next(dataset)
    batch = world_model.preprocess(batch)

    horizons = [0, 5, 15, 50]
    metrics = {}

    with torch.no_grad():
        # Encode observations
        embed = world_model.encoder(batch)

        # Warmup: infer initial posterior latent state at t = warmup - 1
        states, _ = world_model.dynamics.observe(
            embed[:, :warmup], batch["action"][:, :warmup], batch["is_first"][:, :warmup]
        )
        init_state = {k: v[:, -1] for k, v in states.items()}

        def extract_tensor(x):
            if isinstance(x, torch.Tensor):
                return x
            if hasattr(x, "mode"):
                m = x.mode()
                if isinstance(m, torch.Tensor):
                    return m
                elif hasattr(m, "values") and isinstance(m.values, torch.Tensor):
                    return m.values
                return m
            return x

        # ── Horizon 0: Reconstruction from posterior state ─────────────────────
        feat_h0 = world_model.dynamics.get_feat(init_state)
        pred_obs_h0 = {k: extract_tensor(v) for k, v in world_model.heads["decoder"](feat_h0).items()}
        pred_rew_h0 = extract_tensor(world_model.heads["reward"](feat_h0))

        # Ground truth values at warmup boundary (t = warmup - 1)
        last_obs_true = {k: batch[k][:, warmup - 1] for k in pred_obs_h0}
        last_rew_true = batch["reward"][:, warmup - 1]

        # Compute State MSE across all decoded feature keys
        def compute_state_mse(pred_dict, true_dict):
            mses, vars_ = [], []
            for k in pred_dict:
                p = extract_tensor(pred_dict[k])
                t = true_dict[k]
                diff = (p - t) ** 2
                mses.append(diff.mean().item())
                vars_.append(t.var().item() if t.numel() > 1 else 1.0)
            avg_mse = float(np.mean(mses))
            avg_var = float(np.mean(vars_)) if np.mean(vars_) > 1e-6 else 1.0
            return avg_mse, avg_mse / avg_var

        h0_state_mse, h0_state_nmse = compute_state_mse(pred_obs_h0, last_obs_true)
        h0_rew_mse = float(((pred_rew_h0 - last_rew_true) ** 2).mean().item())
        rew_var = float(last_rew_true.var().item()) if last_rew_true.numel() > 1 and last_rew_true.var().item() > 1e-6 else 1.0
        h0_rew_nmse = h0_rew_mse / rew_var

        metrics["horizon_0"] = {
            "steps": 0,
            "type": "reconstruction_posterior",
            "model_state_mse": round(h0_state_mse, 6),
            "model_state_nmse": round(h0_state_nmse, 6),
            "model_reward_mse": round(h0_rew_mse, 6),
            "model_reward_nmse": round(h0_rew_nmse, 6),
            "persistence_state_mse": 0.0,
            "persistence_reward_mse": 0.0,
            "persistence_ratio": 1.0
        }

        # ── Horizons > 0: Open-Loop Rollouts ──────────────────────────────────
        actions = batch["action"][:, warmup:warmup + max_horizon]
        prior = world_model.dynamics.imagine_with_action(actions, init_state)
        feat_all = world_model.dynamics.get_feat(prior)
        pred_obs_all = {k: extract_tensor(v) for k, v in world_model.heads["decoder"](feat_all).items()}
        pred_rew_all = extract_tensor(world_model.heads["reward"](feat_all))

        for h in [5, 15, 50]:
            # Step index in rollout is h - 1 (0-indexed)
            idx = h - 1
            pred_obs_h = {k: v[:, idx] for k, v in pred_obs_all.items()}
            true_obs_h = {k: batch[k][:, warmup + idx] for k in pred_obs_h}

            state_mse, state_nmse = compute_state_mse(pred_obs_h, true_obs_h)
            
            pred_rew_h = pred_rew_all[:, idx]
            true_rew_h = batch["reward"][:, warmup + idx]
            rew_mse = float(((pred_rew_h - true_rew_h) ** 2).mean().item())
            var_r = float(true_rew_h.var().item()) if true_rew_h.var().item() > 1e-6 else 1.0
            rew_nmse = rew_mse / var_r

            # Persistence Baselines: predict last observed state and reward
            persist_state_mse, _ = compute_state_mse(last_obs_true, true_obs_h)
            persist_rew_mse = float(((last_rew_true - true_rew_h) ** 2).mean().item())
            
            p_ratio = state_mse / persist_state_mse if persist_state_mse > 1e-8 else 1.0

            metrics[f"horizon_{h}"] = {
                "steps": h,
                "model_state_mse": round(state_mse, 6),
                "model_state_nmse": round(state_nmse, 6),
                "model_reward_mse": round(rew_mse, 6),
                "model_reward_nmse": round(rew_nmse, 6),
                "persistence_state_mse": round(persist_state_mse, 6),
                "persistence_reward_mse": round(persist_rew_mse, 6),
                "persistence_ratio": round(p_ratio, 4)
            }

    results = {
        "status": "COMPLETED",
        "logdir": str(logdir),
        "device": str(device),
        "capacity_verified": capacity_ok,
        "warmup_steps": warmup,
        "metrics": metrics
    }

    if out_file:
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results

def main():
    parser = argparse.ArgumentParser(description="Upgraded Open-Loop Evaluation v2")
    parser.add_argument("--logdir", type=str, default="logdir/walker_walk_seed0")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--dyn_discrete", type=int, default=None)
    parser.add_argument("--out", type=str, default="open_loop_results_v2.json")
    args = parser.parse_args()

    print("=" * 80)
    print("Open-Loop Prediction Fidelity Evaluation v2 (with Horizon 0 & Persistence Baseline)")
    print("=" * 80)
    res = evaluate_checkpoint(args.logdir, args.warmup, args.out, args.dyn_discrete)
    
    if res.get("status") == "COMPLETED":
        print(f"\nReplay Capacity Check (>= 55 steps): {'PASS' if res['capacity_verified'] else 'FAIL'}")
        print(f"Warmup steps: {res['warmup_steps']}")
        print("\n" + "-" * 88)
        print(f"{'Horizon':<10} | {'Model State MSE':<16} | {'Persist State MSE':<18} | {'Ratio (M/P)':<12} | {'Rew MSE':<10}")
        print("-" * 88)
        for h_key, val in res["metrics"].items():
            h_str = f"H{val['steps']}"
            m_mse = f"{val['model_state_mse']:.4f} (NMSE {val['model_state_nmse']:.2f})"
            p_mse = f"{val['persistence_state_mse']:.4f}"
            ratio = f"{val['persistence_ratio']:.3f}"
            r_mse = f"{val['model_reward_mse']:.4f}"
            print(f"{h_str:<10} | {m_mse:<16} | {p_mse:<18} | {ratio:<12} | {r_mse:<10}")
        print("-" * 88)
        print(f"Saved results to {args.out}")
    else:
        print("Evaluation skipped or failed:", res.get("message"))

if __name__ == "__main__":
    main()
