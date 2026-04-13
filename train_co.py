"""
train_co.py — Joint Caching + Offloading PPO training script.

Usage:
    python train_co.py                        # default 500 episodes
    python train_co.py --episodes 2000 --lr 1e-4
    python train_co.py --resume               # resume from checkpoint
    python train_co.py --eval                  # evaluate only

Live reward curve (during training):
    co_ppo_rewards.npy + co_ppo_reward_curve.png refresh every --plot-every episodes (default 5).
    Tighter refresh:  --plot-every 1
    Separate terminal: watch -n 10 python watch_rewards_plot.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from co_env import JointCacheOffloadEnv  # noqa: E402
from joint_ppo_agent import JointPPOAgent  # noqa: E402

CKPT = "co_ppo_checkpoint.pt"
REWARDS_FILE = "co_ppo_rewards.npy"
PLOT_FILE = "co_ppo_reward_curve.png"

OFFLOAD_NAMES = ["local", "neighbor", "satellite", "drop"]
CACHE_NAMES = ["no-op", "most-useful", "smallest", "pop-density"]


def log(*args, **kw):
    print(*args, **kw, flush=True)


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
def train(args):
    env = JointCacheOffloadEnv(
        grid=(3, 3),
        steps_per_episode=args.steps,
    )
    agent = JointPPOAgent(
        obs_dim=env.obs_dim,
        offload_actions=env.offload_action_dim,
        cache_actions=env.cache_action_dim,
        hidden_dim=args.hidden,
        lr=args.lr,
        entropy_coef=args.ent_coef,
    )

    start_ep = 0
    prev_rewards = []

    if args.resume and os.path.exists(CKPT):
        ckpt = torch.load(CKPT, map_location=agent.device)
        agent.load_state_dict(ckpt["model"])
        agent.optimizer.load_state_dict(ckpt["optimizer"])
        start_ep = ckpt.get("episode", 0)
        if os.path.exists(REWARDS_FILE):
            prev_rewards = np.load(REWARDS_FILE).tolist()
        log(f"Resumed from episode {start_ep}")

    log(f"Joint C+O PPO | episodes {start_ep+1}~{start_ep+args.episodes} | "
        f"obs_dim={env.obs_dim} | hidden={args.hidden} | lr={args.lr}")
    log("=" * 70)

    t0 = time.time()
    rewards = []
    infos_accum = {
        "completed": [], "new_cache_hits": [], "dropped": [],
    }

    for ep_i in range(args.episodes):
        global_ep = start_ep + ep_i + 1
        obs = env.reset()
        agent.reset_hidden()
        ep_reward = 0.0
        ep_info = {"completed": 0, "new_cache_hits": 0, "dropped": 0}

        for _ in range(args.steps):
            off_a, cache_a, lp, val = agent.act(obs)
            next_obs, r, done, info = env.step(off_a, cache_a)
            agent.store(obs, off_a, cache_a, lp, val, r, done)
            obs = next_obs
            ep_reward += r
            for k in ep_info:
                ep_info[k] += info.get(k, 0)
            if done:
                break

        # Final value for GAE bootstrap
        _, _, _, last_val = agent.act(obs, deterministic=True)
        stats = agent.update(last_value=last_val)

        rewards.append(ep_reward)
        for k in infos_accum:
            infos_accum[k].append(ep_info.get(k, 0))

        # Logging: current episode only (no moving average over past K episodes)
        if global_ep % args.log_every == 0:
            log(
                f"Ep {global_ep:4d} | R: {ep_reward:7.2f} | "
                f"completed: {ep_info['completed']:5.0f} | hits: {ep_info['new_cache_hits']:5.0f} | "
                f"drop: {ep_info['dropped']:5.0f} | "
                f"pg={stats.get('pg_loss', 0):.3f} v={stats.get('v_loss', 0):.3f} "
                f"ent={stats.get('entropy', 0):.3f} | {time.time()-t0:.0f}s"
            )

        # Live reward curve (refresh npy + png on a short interval)
        if global_ep % args.plot_every == 0:
            all_r = prev_rewards + rewards
            np.save(REWARDS_FILE, np.array(all_r))
            _plot_curve(all_r, PLOT_FILE, global_ep=global_ep, quiet=not args.plot_log)
            if args.plot_log and global_ep % max(args.plot_every * 5, 25) == 0:
                log(f"  [plot] updated {PLOT_FILE} ({len(all_r)} points)")

        # Checkpoint
        if global_ep % 50 == 0:
            _save(agent, global_ep, prev_rewards + rewards)

    _save(agent, start_ep + args.episodes, prev_rewards + rewards)
    _plot_curve(prev_rewards + rewards, PLOT_FILE, global_ep=start_ep + args.episodes, quiet=False)

    arr = np.array(rewards, dtype=np.float64)
    log(f"\nDone: {len(rewards)} episodes in {time.time()-t0:.0f}s")
    log(
        f"Reward — mean: {arr.mean():.2f}  min: {arr.min():.2f}  max: {arr.max():.2f}  "
        f"last_ep: {arr[-1]:.2f}"
    )


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
def evaluate(args):
    env = JointCacheOffloadEnv(grid=(3, 3), steps_per_episode=args.steps)
    agent = JointPPOAgent(
        obs_dim=env.obs_dim,
        offload_actions=env.offload_action_dim,
        cache_actions=env.cache_action_dim,
        hidden_dim=args.hidden,
    )
    if not os.path.exists(CKPT):
        log("No checkpoint found. Train first.")
        return
    ckpt = torch.load(CKPT, map_location=agent.device)
    agent.load_state_dict(ckpt["model"])
    agent.eval()

    log("Evaluating (deterministic) ...")
    ep_rewards = []
    off_counts = np.zeros(env.offload_action_dim)
    cache_counts = np.zeros(env.cache_action_dim)

    for _ in range(20):
        obs = env.reset()
        agent.reset_hidden()
        ep_r = 0.0
        for _ in range(args.steps):
            off_a, cache_a, _, _ = agent.act(obs, deterministic=True)
            obs, r, done, _ = env.step(off_a, cache_a)
            ep_r += r
            off_counts[off_a] += 1
            cache_counts[cache_a] += 1
            if done:
                break
        ep_rewards.append(ep_r)

    log(f"Eval reward: {np.mean(ep_rewards):.2f} ± {np.std(ep_rewards):.2f}")
    log(f"Offload dist: {dict(zip(OFFLOAD_NAMES, (off_counts / off_counts.sum()).round(3)))}")
    log(f"Cache   dist: {dict(zip(CACHE_NAMES, (cache_counts / cache_counts.sum()).round(3)))}")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _save(agent, episode, all_rewards):
    torch.save({
        "model": agent.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "episode": episode,
    }, CKPT)
    np.save(REWARDS_FILE, np.array(all_rewards))


def _plot_curve(all_rewards, path, global_ep=None, quiet=True):
    """Write reward curve PNG (per-episode only, no moving average)."""
    fig, ax = plt.subplots(figsize=(12, 4))
    n = len(all_rewards)
    ax.plot(all_rewards, color="steelblue", linewidth=0.8, label="episode return")
    ax.set_xlabel("Episode index (0 = start of this run)")
    ax.set_ylabel("Episode reward")
    title = f"Joint C+O PPO — {n} episodes"
    if global_ep is not None:
        title += f" (last global ep {global_ep})"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    if not quiet:
        log(f"Plot saved: {path}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Joint Caching+Offloading PPO")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--steps", type=int, default=50, help="steps per episode")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eval", action="store_true")
    p.add_argument(
        "--plot-every",
        type=int,
        default=5,
        metavar="N",
        help="refresh co_ppo_rewards.npy + co_ppo_reward_curve.png every N episodes (default: 5)",
    )
    p.add_argument(
        "--plot-log",
        action="store_true",
        help="print a line when the live plot is updated (default: silent)",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=10,
        metavar="N",
        help="print one log line every N episodes (each line = that episode only, default: 10)",
    )
    args = p.parse_args()

    if args.plot_every < 1:
        args.plot_every = 1
    if args.log_every < 1:
        args.log_every = 1

    if args.eval:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
