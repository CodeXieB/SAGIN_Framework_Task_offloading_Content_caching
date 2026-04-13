#!/usr/bin/env python3
"""
Re-draw reward curve from co_ppo_rewards.npy while training is running.

Usage (from this directory):
    python watch_rewards_plot.py              # once
    watch -n 10 python watch_rewards_plot.py   # every 10s in another terminal

Or with custom paths:
    python watch_rewards_plot.py --npy my_rewards.npy --png my_curve.png
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NPY = os.path.join(HERE, "co_ppo_rewards.npy")
DEFAULT_PNG = os.path.join(HERE, "co_ppo_reward_curve.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npy", default=DEFAULT_NPY, help="path to rewards .npy")
    p.add_argument("--png", default=DEFAULT_PNG, help="output PNG path")
    args = p.parse_args()

    if not os.path.isfile(args.npy):
        print("No file yet:", args.npy, file=sys.stderr)
        sys.exit(1)

    data = np.load(args.npy)
    if data.ndim != 1:
        data = data.ravel()
    n = len(data)
    if n == 0:
        print("Empty rewards file.", file=sys.stderr)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(data, color="steelblue", linewidth=0.8, label="episode return")
    ax.set_xlabel("Episode index")
    ax.set_ylabel("Episode reward")
    ax.set_title(f"Joint C+O PPO — {n} episodes (from {os.path.basename(args.npy)})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.png, dpi=100)
    plt.close(fig)
    print("Updated:", args.png, "| points:", n)


if __name__ == "__main__":
    main()
