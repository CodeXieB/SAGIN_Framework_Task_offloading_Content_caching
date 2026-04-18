"""
Shared PPO training/evaluation utilities for baseline and safe-reward variants.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from typing import Dict, List

import numpy as np
import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional plotting dependency
    matplotlib = None
    plt = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from co_env import JointCacheOffloadEnv  # noqa: E402
from joint_ppo_agent import JointPPOAgent  # noqa: E402

OFFLOAD_NAMES = ["local", "neighbor", "satellite", "drop"]
CACHE_NAMES = ["no-op", "most-useful", "smallest", "pop-density"]


def log(*args, **kw):
    print(*args, **kw, flush=True)


def build_parser(description: str, default_artifact_dir: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--steps", type=int, default=50, help="steps per episode")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--artifact-dir", default=default_artifact_dir)
    p.add_argument("--plot-every", type=int, default=5, metavar="N")
    p.add_argument("--plot-log", action="store_true")
    p.add_argument("--log-every", type=int, default=10, metavar="N")
    p.add_argument("--smooth-window", type=int, default=20, metavar="N")

    # Environment parameters kept explicit so later experiments can reuse the same core.
    p.add_argument("--grid-x", type=int, default=3)
    p.add_argument("--grid-y", type=int, default=3)
    p.add_argument("--duration", type=int, default=300)
    p.add_argument("--cache-size", type=int, default=40)
    p.add_argument("--compute-power-uav", type=int, default=25)
    p.add_argument("--compute-power-sat", type=int, default=200)
    p.add_argument("--energy", type=int, default=80000)
    p.add_argument("--max-queue", type=int, default=15)
    p.add_argument("--num-sats", type=int, default=2)
    p.add_argument("--num-iot-per-region", type=int, default=20)
    p.add_argument("--max-active-iot", type=int, default=10)
    p.add_argument("--ofdm-slots", type=int, default=6)

    # Safe reward parameters. Baseline ignores them, A/B use them directly.
    p.add_argument("--alpha1", type=float, default=1.0)
    p.add_argument("--alpha2", type=float, default=0.3)
    p.add_argument("--alpha3", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=1.0)
    p.add_argument("--beta2", type=float, default=1.0)
    p.add_argument("--beta3", type=float, default=1.0)
    p.add_argument("--lambda0", type=float, default=0.2)
    p.add_argument("--k", type=float, default=1.0)
    p.add_argument("--eta", type=float, default=1.5)
    p.add_argument("--gamma", type=float, default=0.25)
    p.add_argument("--cmax", type=float, default=1.8)
    return p


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.plot_every < 1:
        args.plot_every = 1
    if args.log_every < 1:
        args.log_every = 1
    if args.smooth_window < 1:
        args.smooth_window = 1
    return args


def _slugify_experiment_name(experiment_name: str) -> str:
    slug = experiment_name.strip().lower().replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in slug)


def _resolve_plot_path(artifact_dir: str, experiment_name: str, resume: bool) -> str:
    slug = _slugify_experiment_name(experiment_name)
    pattern = re.compile(rf"^{re.escape(slug)}_reward_curve_(\d+)\.png$")
    existing = []
    for name in os.listdir(artifact_dir):
        match = pattern.match(name)
        if match:
            existing.append(int(match.group(1)))

    if resume and existing:
        idx = max(existing)
    else:
        idx = (max(existing) + 1) if existing else 1
    return os.path.join(artifact_dir, f"{slug}_reward_curve_{idx:03d}.png")


def _artifact_paths(artifact_dir: str, experiment_name: str, resume: bool) -> Dict[str, str]:
    os.makedirs(artifact_dir, exist_ok=True)
    return {
        "dir": artifact_dir,
        "ckpt": os.path.join(artifact_dir, "checkpoint.pt"),
        "rewards": os.path.join(artifact_dir, "rewards.npy"),
        "metrics": os.path.join(artifact_dir, "metrics.npz"),
        "plot": _resolve_plot_path(artifact_dir, experiment_name, resume),
    }


def _make_env(args: argparse.Namespace, reward_mode: str) -> JointCacheOffloadEnv:
    return JointCacheOffloadEnv(
        grid=(args.grid_x, args.grid_y),
        duration=args.duration,
        cache_size=args.cache_size,
        compute_power_uav=args.compute_power_uav,
        compute_power_sat=args.compute_power_sat,
        energy=args.energy,
        max_queue=args.max_queue,
        num_sats=args.num_sats,
        num_iot_per_region=args.num_iot_per_region,
        max_active_iot=args.max_active_iot,
        ofdm_slots=args.ofdm_slots,
        steps_per_episode=args.steps,
        reward_mode=reward_mode,
        alpha1=args.alpha1,
        alpha2=args.alpha2,
        alpha3=args.alpha3,
        beta1=args.beta1,
        beta2=args.beta2,
        beta3=args.beta3,
        lambda0=args.lambda0,
        k=args.k,
        eta=args.eta,
        gamma=args.gamma,
        cmax=args.cmax,
    )


def _make_agent(args: argparse.Namespace, env: JointCacheOffloadEnv, training: bool = True) -> JointPPOAgent:
    kwargs = dict(
        obs_dim=env.obs_dim,
        offload_actions=env.offload_action_dim,
        cache_actions=env.cache_action_dim,
        hidden_dim=args.hidden,
    )
    if training:
        kwargs.update(lr=args.lr, entropy_coef=args.ent_coef)
    return JointPPOAgent(**kwargs)


def _moving_average(values: List[float], window: int):
    if window <= 1 or len(values) < window:
        return None, None
    kernel = np.ones(window, dtype=np.float64) / float(window)
    smoothed = np.convolve(np.asarray(values, dtype=np.float64), kernel, mode="valid")
    x = np.arange(window - 1, window - 1 + len(smoothed))
    return x, smoothed


def _plot_curve(all_rewards: List[float], path: str, title: str, global_ep: int | None = None,
                quiet: bool = True, smooth_window: int = 20) -> None:
    if plt is None:
        if not quiet:
            log("Plot skipped: matplotlib is not installed.")
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    n = len(all_rewards)
    ax.plot(all_rewards, color="steelblue", linewidth=0.8, alpha=0.35, label="episode return")
    ma_x, ma_y = _moving_average(all_rewards, smooth_window)
    if ma_y is not None:
        ax.plot(ma_x, ma_y, color="darkorange", linewidth=2.0,
                label=f"moving average ({smooth_window})")
    ax.set_xlabel("Episode index")
    ax.set_ylabel("Episode reward")
    full_title = f"{title} - {n} episodes"
    if global_ep is not None:
        full_title += f" (last ep {global_ep})"
    ax.set_title(full_title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    if not quiet:
        log(f"Plot saved: {path}")


def _save(agent: JointPPOAgent, episode: int, metrics: Dict[str, List[float]], paths: Dict[str, str]) -> None:
    torch.save({
        "model": agent.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "episode": episode,
    }, paths["ckpt"])
    np.save(paths["rewards"], np.asarray(metrics["episode_rewards"], dtype=np.float64))
    np.savez(paths["metrics"], **{k: np.asarray(v) for k, v in metrics.items()})


def train_experiment(args: argparse.Namespace, reward_mode: str, experiment_name: str) -> None:
    paths = _artifact_paths(args.artifact_dir, experiment_name, args.resume)
    env = _make_env(args, reward_mode)
    agent = _make_agent(args, env, training=True)

    start_ep = 0
    prev_metrics = None
    metrics = {
        "episode_rewards": [],
        "episode_completed": [],
        "episode_hits": [],
        "episode_dropped": [],
        "episode_violation_steps": [],
        "episode_avg_csafe": [],
        "episode_avg_rperf": [],
        "episode_avg_energy_ratio": [],
        "episode_avg_queue_ratio": [],
        "episode_avg_failed_offloads": [],
    }

    if args.resume and os.path.exists(paths["ckpt"]):
        ckpt = torch.load(paths["ckpt"], map_location=agent.device)
        agent.load_state_dict(ckpt["model"])
        agent.optimizer.load_state_dict(ckpt["optimizer"])
        start_ep = ckpt.get("episode", 0)
        if os.path.exists(paths["metrics"]):
            prev_metrics = np.load(paths["metrics"], allow_pickle=False)
            for k in metrics:
                if k in prev_metrics:
                    metrics[k] = prev_metrics[k].tolist()
        log(f"Resumed {experiment_name} from episode {start_ep}")

    log(
        f"{experiment_name} | reward={reward_mode} | episodes {start_ep + 1}~{start_ep + args.episodes} | "
        f"obs_dim={env.obs_dim} | hidden={args.hidden} | lr={args.lr}"
    )
    log("=" * 80)

    t0 = time.time()

    for ep_i in range(args.episodes):
        global_ep = start_ep + ep_i + 1
        obs = env.reset()
        agent.reset_hidden()
        ep_reward = 0.0
        step_count = 0
        ep_sums = {
            "completed": 0.0,
            "new_cache_hits": 0.0,
            "dropped": 0.0,
            "constraint_violated": 0.0,
            "c_safe": 0.0,
            "r_perf": 0.0,
            "energy_ratio": 0.0,
            "queue_ratio": 0.0,
            "failed_offloads": 0.0,
        }

        for _ in range(args.steps):
            off_a, cache_a, lp, val = agent.act(obs)
            next_obs, r, done, info = env.step(off_a, cache_a)
            agent.store(obs, off_a, cache_a, lp, val, r, done)
            obs = next_obs
            ep_reward += r
            step_count += 1
            for k in ep_sums:
                ep_sums[k] += float(info.get(k, 0.0))
            if done:
                break

        _, _, _, last_val = agent.act(obs, deterministic=True)
        stats = agent.update(last_value=last_val)

        denom = float(max(step_count, 1))
        metrics["episode_rewards"].append(ep_reward)
        metrics["episode_completed"].append(ep_sums["completed"])
        metrics["episode_hits"].append(ep_sums["new_cache_hits"])
        metrics["episode_dropped"].append(ep_sums["dropped"])
        metrics["episode_violation_steps"].append(ep_sums["constraint_violated"])
        metrics["episode_avg_csafe"].append(ep_sums["c_safe"] / denom)
        metrics["episode_avg_rperf"].append(ep_sums["r_perf"] / denom)
        metrics["episode_avg_energy_ratio"].append(ep_sums["energy_ratio"] / denom)
        metrics["episode_avg_queue_ratio"].append(ep_sums["queue_ratio"] / denom)
        metrics["episode_avg_failed_offloads"].append(ep_sums["failed_offloads"] / denom)

        if global_ep % args.log_every == 0:
            log(
                f"Ep {global_ep:4d} | R: {ep_reward:7.2f} | comp: {ep_sums['completed']:5.0f} | "
                f"hits: {ep_sums['new_cache_hits']:5.0f} | drop: {ep_sums['dropped']:5.0f} | "
                f"vio: {ep_sums['constraint_violated']:4.0f} | csafe: {ep_sums['c_safe'] / denom:.3f} | "
                f"pg={stats.get('pg_loss', 0):.3f} v={stats.get('v_loss', 0):.3f} "
                f"ent={stats.get('entropy', 0):.3f} | {time.time() - t0:.0f}s"
            )

        if global_ep % args.plot_every == 0:
            _save(agent, global_ep, metrics, paths)
            _plot_curve(
                metrics["episode_rewards"],
                paths["plot"],
                experiment_name,
                global_ep=global_ep,
                quiet=not args.plot_log,
                smooth_window=args.smooth_window,
            )

        if global_ep % 50 == 0:
            _save(agent, global_ep, metrics, paths)

    final_ep = start_ep + args.episodes
    _save(agent, final_ep, metrics, paths)
    _plot_curve(
        metrics["episode_rewards"],
        paths["plot"],
        experiment_name,
        global_ep=final_ep,
        quiet=False,
        smooth_window=args.smooth_window,
    )

    arr = np.asarray(metrics["episode_rewards"], dtype=np.float64)
    log(f"\nDone: {len(arr)} episodes in {time.time() - t0:.0f}s")
    log(
        f"Reward mean: {arr.mean():.2f} | min: {arr.min():.2f} | max: {arr.max():.2f} | "
        f"last_ep: {arr[-1]:.2f}"
    )


def evaluate_experiment(args: argparse.Namespace, reward_mode: str, experiment_name: str) -> None:
    paths = _artifact_paths(args.artifact_dir, experiment_name, args.resume)
    env = _make_env(args, reward_mode)
    agent = _make_agent(args, env, training=False)

    if not os.path.exists(paths["ckpt"]):
        log(f"No checkpoint found for {experiment_name}. Train first.")
        return

    ckpt = torch.load(paths["ckpt"], map_location=agent.device)
    agent.load_state_dict(ckpt["model"])
    agent.eval()

    log(f"Evaluating {experiment_name} (deterministic) ...")
    ep_rewards = []
    off_counts = np.zeros(env.offload_action_dim)
    cache_counts = np.zeros(env.cache_action_dim)
    violation_steps = []
    avg_csafe = []

    for _ in range(20):
        obs = env.reset()
        agent.reset_hidden()
        ep_r = 0.0
        ep_vio = 0.0
        ep_csafe = 0.0
        steps = 0
        for _ in range(args.steps):
            off_a, cache_a, _, _ = agent.act(obs, deterministic=True)
            obs, r, done, info = env.step(off_a, cache_a)
            ep_r += r
            ep_vio += float(info.get("constraint_violated", 0.0))
            ep_csafe += float(info.get("c_safe", 0.0))
            steps += 1
            off_counts[off_a] += 1
            cache_counts[cache_a] += 1
            if done:
                break
        ep_rewards.append(ep_r)
        violation_steps.append(ep_vio)
        avg_csafe.append(ep_csafe / max(steps, 1))

    log(f"Eval reward: {np.mean(ep_rewards):.2f} +/- {np.std(ep_rewards):.2f}")
    log(f"Avg violation steps: {np.mean(violation_steps):.2f}")
    log(f"Avg c_safe: {np.mean(avg_csafe):.3f}")
    log(f"Offload dist: {dict(zip(OFFLOAD_NAMES, (off_counts / off_counts.sum()).round(3)))}")
    log(f"Cache   dist: {dict(zip(CACHE_NAMES, (cache_counts / cache_counts.sum()).round(3)))}")


def main_for_experiment(default_reward_mode: str, experiment_name: str, default_artifact_dir: str) -> None:
    parser = build_parser(experiment_name, default_artifact_dir)
    args = normalize_args(parser.parse_args())
    if args.eval:
        evaluate_experiment(args, default_reward_mode, experiment_name)
    else:
        train_experiment(args, default_reward_mode, experiment_name)
