"""
Create comparison plots for Baseline / Safe-A / Safe-B / Safe-C experiments.

Expected inputs:
    artifacts/<scheme>/metrics.npz

Outputs (default):
    artifacts/summary/
        reward_curves.png
        performance_bars.png
        safety_bars.png
        failed_offloads_bar.png
        perf_safety_tradeoff.png
        safe_ab_zoom_curves.png
        safe_ab_delta_bars.png
        safe_ab_stability.png
        summary.csv

Usage:
    python plot_experiment_comparison.py
    python plot_experiment_comparison.py --tail-n 50
    python plot_experiment_comparison.py --schemes baseline safe_b safe_c
"""
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "matplotlib is required to draw comparison figures. "
        "Install it in the current environment first."
    ) from exc


@dataclass
class SchemeData:
    key: str
    label: str
    color: str
    metrics: Dict[str, np.ndarray]


SCHEME_META = {
    "baseline": ("Baseline", "#1f77b4"),
    "safe_a": ("Safe-A", "#d62728"),
    "safe_b": ("Safe-B", "#2ca02c"),
    "safe_c": ("Safe-C", "#ff7f0e"),
}

METRIC_KEYS = [
    "episode_rewards",
    "episode_completed",
    "episode_hits",
    "episode_dropped",
    "episode_violation_steps",
    "episode_avg_csafe",
    "episode_avg_rperf",
    "episode_avg_failed_offloads",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot experiment comparison figures.")
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="root artifacts directory containing baseline/safe_* folders",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("artifacts", "summary"),
        help="directory to store generated figures and summary.csv",
    )
    parser.add_argument(
        "--schemes",
        nargs="+",
        default=["baseline", "safe_a", "safe_b", "safe_c"],
        help="schemes to include in plots",
    )
    parser.add_argument(
        "--tail-n",
        type=int,
        default=50,
        help="number of trailing episodes used for bar/scatter summaries",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=20,
        help="moving-average window for reward curves; <=1 disables smoothing",
    )
    return parser.parse_args()


def moving_average(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    if window <= 1 or len(values) < window:
        return None, None
    kernel = np.ones(window, dtype=np.float64) / float(window)
    smoothed = np.convolve(values.astype(np.float64), kernel, mode="valid")
    x = np.arange(window - 1, window - 1 + len(smoothed))
    return x, smoothed


def load_scheme(artifacts_dir: Path, key: str) -> SchemeData | None:
    metrics_path = artifacts_dir / key / "metrics.npz"
    if not metrics_path.exists():
        print(f"[skip] missing metrics: {metrics_path}")
        return None
    raw = np.load(metrics_path)
    label, color = SCHEME_META.get(key, (key, "#7f7f7f"))
    metrics = {name: raw[name].astype(float) for name in METRIC_KEYS if name in raw}
    return SchemeData(key=key, label=label, color=color, metrics=metrics)


def tail_stats(values: np.ndarray, tail_n: int) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    tail = arr[-tail_n:] if len(arr) >= tail_n else arr
    return float(tail.mean()), float(tail.std())


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_reward_curves(schemes: List[SchemeData], output_dir: Path, smooth_window: int) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for scheme in schemes:
        rewards = scheme.metrics["episode_rewards"]
        ax.plot(
            rewards,
            color=scheme.color,
            alpha=0.18,
            linewidth=0.8,
        )
        ma_x, ma_y = moving_average(rewards, smooth_window)
        if ma_y is None:
            ax.plot(rewards, color=scheme.color, linewidth=1.8, label=scheme.label)
        else:
            ax.plot(ma_x, ma_y, color=scheme.color, linewidth=2.2, label=scheme.label)

    ax.set_title("Training Reward Curves")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "reward_curves.png", dpi=150)
    plt.close(fig)


def plot_bar_pair(
    schemes: List[SchemeData],
    output_dir: Path,
    tail_n: int,
    left_metric: str,
    right_metric: str,
    left_title: str,
    right_title: str,
    file_name: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = [s.label for s in schemes]
    colors = [s.color for s in schemes]
    x = np.arange(len(schemes))

    for ax, metric, title in zip(axes, [left_metric, right_metric], [left_title, right_title]):
        means = []
        stds = []
        for scheme in schemes:
            mean, std = tail_stats(scheme.metrics[metric], tail_n)
            means.append(mean)
            stds.append(std)
        ax.bar(x, means, yerr=stds, color=colors, capsize=4, alpha=0.9)
        ax.set_xticks(x, labels)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(f"Tail-{tail_n} Episode Comparison", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / file_name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_failed_offloads(schemes: List[SchemeData], output_dir: Path, tail_n: int) -> None:
    labels = [s.label for s in schemes]
    colors = [s.color for s in schemes]
    x = np.arange(len(schemes))
    means = []
    stds = []
    for scheme in schemes:
        mean, std = tail_stats(scheme.metrics["episode_avg_failed_offloads"], tail_n)
        means.append(mean)
        stds.append(std)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x, means, yerr=stds, color=colors, capsize=4, alpha=0.9)
    ax.set_xticks(x, labels)
    ax.set_title(f"Failed Offloads (Tail-{tail_n} Episodes)")
    ax.set_ylabel("Average Failed Offloads")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "failed_offloads_bar.png", dpi=150)
    plt.close(fig)


def plot_tradeoff_scatter(schemes: List[SchemeData], output_dir: Path, tail_n: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for scheme in schemes:
        x_mean, x_std = tail_stats(scheme.metrics["episode_avg_csafe"], tail_n)
        y_mean, y_std = tail_stats(scheme.metrics["episode_completed"], tail_n)
        ax.scatter(x_mean, y_mean, color=scheme.color, s=120, label=scheme.label)
        ax.errorbar(x_mean, y_mean, xerr=x_std, yerr=y_std, color=scheme.color, alpha=0.45, capsize=3)
        ax.annotate(scheme.label, (x_mean, y_mean), textcoords="offset points", xytext=(6, 6))

    ax.set_title(f"Performance-Safety Tradeoff (Tail-{tail_n} Episodes)")
    ax.set_xlabel("Average c_safe")
    ax.set_ylabel("Completed Tasks")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "perf_safety_tradeoff.png", dpi=150)
    plt.close(fig)


def _find_scheme(schemes: List[SchemeData], key: str) -> SchemeData | None:
    for scheme in schemes:
        if scheme.key == key:
            return scheme
    return None


def plot_safe_ab_zoom_curves(schemes: List[SchemeData], output_dir: Path, smooth_window: int) -> None:
    safe_a = _find_scheme(schemes, "safe_a")
    safe_b = _find_scheme(schemes, "safe_b")
    if safe_a is None or safe_b is None:
        return

    plots = [
        ("episode_rewards", "Episode Reward"),
        ("episode_completed", "Completed Tasks"),
        ("episode_violation_steps", "Violation Steps"),
        ("episode_avg_csafe", "Average c_safe"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, (metric, title) in zip(axes.ravel(), plots):
        for scheme in [safe_a, safe_b]:
            values = scheme.metrics[metric]
            ma_x, ma_y = moving_average(values, smooth_window)
            if ma_y is None:
                ax.plot(values, color=scheme.color, linewidth=1.6, label=scheme.label)
            else:
                ax.plot(ma_x, ma_y, color=scheme.color, linewidth=2.0, label=scheme.label)
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.grid(True, alpha=0.25)
        ax.legend()

    fig.suptitle("Safe-A vs Safe-B Zoomed Training Curves", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "safe_ab_zoom_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _relative_gain(a_value: float, b_value: float, higher_is_better: bool) -> float:
    """Return B-vs-A relative improvement in percent; positive means B is better."""
    if higher_is_better:
        diff = b_value - a_value
    else:
        diff = a_value - b_value
    denom = max(abs(a_value), 1e-9)
    return 100.0 * diff / denom


def plot_safe_ab_delta_bars(schemes: List[SchemeData], output_dir: Path, tail_n: int) -> None:
    safe_a = _find_scheme(schemes, "safe_a")
    safe_b = _find_scheme(schemes, "safe_b")
    if safe_a is None or safe_b is None:
        return

    comparisons = [
        ("episode_rewards", "Reward", True),
        ("episode_completed", "Completed", True),
        ("episode_hits", "Hits", True),
        ("episode_dropped", "Dropped", False),
        ("episode_violation_steps", "Violations", False),
        ("episode_avg_csafe", "c_safe", False),
        ("episode_avg_failed_offloads", "Failed Offloads", False),
    ]

    labels = []
    gains = []
    for metric, label, higher_is_better in comparisons:
        a_mean, _ = tail_stats(safe_a.metrics[metric], tail_n)
        b_mean, _ = tail_stats(safe_b.metrics[metric], tail_n)
        labels.append(label)
        gains.append(_relative_gain(a_mean, b_mean, higher_is_better))

    colors = ["#2ca02c" if value >= 0 else "#d62728" for value in gains]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x, gains, color=colors, alpha=0.9)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Relative improvement of Safe-B over Safe-A (%)")
    ax.set_title(f"Safe-B vs Safe-A Tail-{tail_n} Relative Improvement")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "safe_ab_delta_bars.png", dpi=150)
    plt.close(fig)


def plot_safe_ab_stability(schemes: List[SchemeData], output_dir: Path, tail_n: int) -> None:
    safe_a = _find_scheme(schemes, "safe_a")
    safe_b = _find_scheme(schemes, "safe_b")
    if safe_a is None or safe_b is None:
        return

    metrics = [
        ("episode_rewards", "Reward Std"),
        ("episode_completed", "Completed Std"),
        ("episode_violation_steps", "Violation Std"),
        ("episode_avg_csafe", "c_safe Std"),
    ]
    labels = [label for _, label in metrics]
    x = np.arange(len(labels))
    width = 0.36

    a_stds = [tail_stats(safe_a.metrics[metric], tail_n)[1] for metric, _ in metrics]
    b_stds = [tail_stats(safe_b.metrics[metric], tail_n)[1] for metric, _ in metrics]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(x - width / 2, a_stds, width, label=safe_a.label, color=safe_a.color, alpha=0.85)
    ax.bar(x + width / 2, b_stds, width, label=safe_b.label, color=safe_b.color, alpha=0.85)
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel("Tail standard deviation")
    ax.set_title(f"Safe-A vs Safe-B Stability (Tail-{tail_n})")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "safe_ab_stability.png", dpi=150)
    plt.close(fig)


def write_summary_csv(schemes: List[SchemeData], output_dir: Path, tail_n: int) -> None:
    rows = []
    metrics_to_export = [
        "episode_rewards",
        "episode_completed",
        "episode_hits",
        "episode_dropped",
        "episode_violation_steps",
        "episode_avg_csafe",
        "episode_avg_rperf",
        "episode_avg_failed_offloads",
    ]

    for scheme in schemes:
        row = {"scheme": scheme.label, "episodes": len(scheme.metrics["episode_rewards"])}
        for metric in metrics_to_export:
            mean, std = tail_stats(scheme.metrics[metric], tail_n)
            row[f"{metric}_tail_mean"] = mean
            row[f"{metric}_tail_std"] = std
        rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else ["scheme"]
    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)

    schemes: List[SchemeData] = []
    for key in args.schemes:
        scheme = load_scheme(artifacts_dir, key)
        if scheme is not None:
            schemes.append(scheme)

    if not schemes:
        raise SystemExit("No valid metrics.npz files were found. Nothing to plot.")

    plot_reward_curves(schemes, output_dir, args.smooth_window)
    plot_bar_pair(
        schemes,
        output_dir,
        args.tail_n,
        left_metric="episode_completed",
        right_metric="episode_hits",
        left_title="Completed Tasks",
        right_title="Cache Hits",
        file_name="performance_bars.png",
    )
    plot_bar_pair(
        schemes,
        output_dir,
        args.tail_n,
        left_metric="episode_violation_steps",
        right_metric="episode_avg_csafe",
        left_title="Violation Steps",
        right_title="Average c_safe",
        file_name="safety_bars.png",
    )
    plot_failed_offloads(schemes, output_dir, args.tail_n)
    plot_tradeoff_scatter(schemes, output_dir, args.tail_n)
    plot_safe_ab_zoom_curves(schemes, output_dir, args.smooth_window)
    plot_safe_ab_delta_bars(schemes, output_dir, args.tail_n)
    plot_safe_ab_stability(schemes, output_dir, args.tail_n)
    write_summary_csv(schemes, output_dir, args.tail_n)

    print(f"Saved comparison figures to: {output_dir}")


if __name__ == "__main__":
    main()
