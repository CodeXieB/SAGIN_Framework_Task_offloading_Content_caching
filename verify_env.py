"""
verify_env.py — Quick sanity checks for co_env learnability.

Tests:
  1. Fixed-policy sweep: all 16 (offload x cache) combos
  2. Random baseline
  3. Verdict: is there reward separation?
  4. Quick PPO convergence (30 episodes)

Usage:
    python verify_env.py
"""
from __future__ import annotations

import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from co_env import JointCacheOffloadEnv

OFFLOAD_NAMES = ["local", "neighbor", "satellite", "drop"]
CACHE_NAMES = ["no-op", "most-useful", "smallest", "pop-density"]

EPISODES = 10
STEPS = 50


def log(*a, **kw):
    print(*a, **kw, flush=True)


def run_fixed(env, off_a, cache_a, episodes=EPISODES, steps=STEPS):
    rewards = []
    total_info = {"completed": 0, "new_cache_hits": 0, "dropped": 0, "failed_offloads": 0}
    for _ in range(episodes):
        env.reset()
        ep_r = 0.0
        for _ in range(steps):
            _, r, done, info = env.step(off_a, cache_a)
            ep_r += r
            for k in total_info:
                total_info[k] += info.get(k, 0)
            if done:
                break
        rewards.append(ep_r)
    n = float(episodes)
    avg_info = {k: v / n for k, v in total_info.items()}
    return np.mean(rewards), np.std(rewards), avg_info


def run_random(env, episodes=EPISODES, steps=STEPS):
    rewards = []
    for _ in range(episodes):
        env.reset()
        ep_r = 0.0
        for _ in range(steps):
            off_a = np.random.randint(0, env.offload_action_dim)
            cache_a = np.random.randint(0, env.cache_action_dim)
            _, r, done, _ = env.step(off_a, cache_a)
            ep_r += r
            if done:
                break
        rewards.append(ep_r)
    return np.mean(rewards), np.std(rewards)


def main():
    env = JointCacheOffloadEnv(grid=(3, 3), steps_per_episode=STEPS)
    log("=" * 72)
    log("Environment Learnability Verification (post-fix)")
    log("=" * 72)

    # --- Test 1: Fixed-policy sweep ---
    log("\n[Test 1] Fixed-policy sweep (all 16 combos, %d ep x %d steps)" % (EPISODES, STEPS))
    log("-" * 72)
    results = []
    for off_a in range(env.offload_action_dim):
        for cache_a in range(env.cache_action_dim):
            mean_r, std_r, avg_info = run_fixed(env, off_a, cache_a)
            results.append((off_a, cache_a, mean_r, std_r, avg_info))
            log("  off=%-10s cache=%-12s  R=%7.2f +/-%5.2f  comp=%5.0f hits=%5.0f drop=%5.0f fail=%5.0f" % (
                OFFLOAD_NAMES[off_a], CACHE_NAMES[cache_a],
                mean_r, std_r,
                avg_info["completed"], avg_info["new_cache_hits"],
                avg_info["dropped"], avg_info["failed_offloads"]))

    results.sort(key=lambda x: x[2], reverse=True)
    best = results[0]
    worst = results[-1]
    log("\n  BEST:  off=%-10s cache=%-12s  R=%.2f" % (
        OFFLOAD_NAMES[best[0]], CACHE_NAMES[best[1]], best[2]))
    log("  WORST: off=%-10s cache=%-12s  R=%.2f" % (
        OFFLOAD_NAMES[worst[0]], CACHE_NAMES[worst[1]], worst[2]))
    spread = best[2] - worst[2]
    log("  SPREAD = %.2f" % spread)

    # --- Test 2: Random ---
    log("\n[Test 2] Random policy (%d ep)" % EPISODES)
    log("-" * 72)
    rand_mean, rand_std = run_random(env)
    log("  RANDOM:     R = %.2f +/- %.2f" % (rand_mean, rand_std))
    log("  BEST FIXED: R = %.2f" % best[2])
    gap = best[2] - rand_mean
    log("  GAP = %.2f (%.1f%%)" % (gap, 100 * gap / max(abs(rand_mean), 0.01)))

    # --- Test 3: Verdict ---
    log("\n[Test 3] Verdict")
    log("-" * 72)
    if spread < 0.5:
        log("  BAD: All combos give similar reward (spread=%.2f). Actions don't matter." % spread)
    elif gap < 1.0:
        log("  WEAK: Best fixed barely beats random (gap=%.2f)." % gap)
    else:
        log("  OK: Clear separation. Best fixed beats random by %.2f." % gap)
        log("  PPO should learn this.")

    # --- Test 4: Quick PPO ---
    log("\n[Test 4] Quick PPO convergence (30 ep)")
    log("-" * 72)
    try:
        from joint_ppo_agent import JointPPOAgent
        agent = JointPPOAgent(obs_dim=env.obs_dim, lr=1e-3, entropy_coef=0.02)
        ep_rewards = []
        for ep in range(30):
            obs = env.reset()
            agent.reset_hidden()
            ep_r = 0.0
            for _ in range(STEPS):
                off_a, cache_a, lp, val = agent.act(obs)
                obs2, r, done, _ = env.step(off_a, cache_a)
                agent.store(obs, off_a, cache_a, lp, val, r, done)
                obs = obs2
                ep_r += r
                if done:
                    break
            _, _, _, last_val = agent.act(obs, deterministic=True)
            agent.update(last_value=last_val)
            ep_rewards.append(ep_r)

        first5 = np.mean(ep_rewards[:5])
        last5 = np.mean(ep_rewards[-5:])
        log("  First 5 avg: %.2f" % first5)
        log("  Last  5 avg: %.2f" % last5)
        delta = last5 - first5
        if delta > 2.0:
            log("  LEARNING: +%.2f improvement in 30 episodes." % delta)
        elif delta > 0:
            log("  SLIGHT: +%.2f. May need more episodes or tuning." % delta)
        else:
            log("  NO IMPROVEMENT (%.2f). Check reward scale / lr." % delta)
    except Exception as e:
        log("  PPO test failed: %s" % str(e))

    log("\n" + "=" * 72)
    log("Verification complete.")


if __name__ == "__main__":
    main()
