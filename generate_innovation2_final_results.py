"""
Generate final tables and figures for Innovation 2 experiments.

Main comparison:
    Safe-C-800                 -> no curriculum
    Curriculum-Safe-Target     -> learning-progress curriculum

Outputs are saved under:
    artifacts/generated_results/<timestamp>/

The script only reads existing metrics.npz files.

Example:
    python generate_innovation2_final_results.py
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:  # pragma: no cover
    plt = None
    HAS_MPL = False


SCHEMES: List[Tuple[str, str, str]] = [
    ("safe_c_800", "Safe-C-800", "#ff7f0e"),
    ("curriculum_safe_target_final", "Curriculum-Safe-800", "#9467bd"),
]

METRICS: List[Tuple[str, str, str]] = [
    ("episode_rewards", "Reward", "higher"),
    ("episode_completed", "Completed Tasks", "higher"),
    ("episode_hits", "Cache Hits", "higher"),
    ("episode_dropped", "Dropped Tasks", "lower"),
    ("episode_violation_steps", "Violation Steps", "lower"),
    ("episode_avg_csafe", "Average Safety Cost", "lower"),
    ("episode_avg_rperf", "Average Performance Reward", "higher"),
    ("episode_avg_failed_offloads", "Average Failed Offloads", "lower"),
]

CURRICULUM_LEVEL_NAMES = ["Easy", "Medium", "Target", "Hard"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fixed final results for Innovation 2.")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--tail-n", type=int, default=50)
    parser.add_argument("--smooth-window", type=int, default=20)
    parser.add_argument("--run-id", default=None, help="Optional fixed output directory name.")
    return parser.parse_args()


def tail_stats(values: np.ndarray, tail_n: int) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    tail = values[-tail_n:] if values.size >= tail_n else values
    return float(np.nanmean(tail)), float(np.nanstd(tail))


def moving_average(values: np.ndarray, window: int) -> Tuple[np.ndarray | None, np.ndarray | None]:
    if window <= 1 or values.size < window:
        return None, None
    kernel = np.ones(window, dtype=np.float64) / float(window)
    smoothed = np.convolve(values.astype(np.float64), kernel, mode="valid")
    x = np.arange(window - 1, window - 1 + smoothed.size)
    return x, smoothed


def load_all(artifact_root: Path, tail_n: int) -> Tuple[List[Dict[str, float | str]], Dict[str, Dict[str, np.ndarray]]]:
    rows: List[Dict[str, float | str]] = []
    all_metrics: Dict[str, Dict[str, np.ndarray]] = {}
    for scheme_key, scheme_label, _ in SCHEMES:
        metrics_path = artifact_root / scheme_key / "metrics.npz"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
        data_npz = np.load(metrics_path, allow_pickle=False)
        data = {key: data_npz[key] for key in data_npz.files}
        all_metrics[scheme_key] = data

        row: Dict[str, float | str] = {
            "scheme": scheme_key,
            "label": scheme_label,
            "episodes": int(len(data["episode_rewards"])),
        }
        for metric_key, _, _ in METRICS:
            if metric_key not in data:
                raise KeyError(f"{metrics_path} is missing metric: {metric_key}")
            mean, std = tail_stats(data[metric_key], tail_n)
            row[f"{metric_key}_mean"] = mean
            row[f"{metric_key}_std"] = std
        rows.append(row)
    return rows, all_metrics


def write_summary(rows: List[Dict[str, float | str]], output_dir: Path, tail_n: int) -> Tuple[Path, Path]:
    csv_path = output_dir / "innovation2_final_summary.csv"
    fieldnames = ["scheme", "label", "episodes"]
    for metric_key, _, _ in METRICS:
        fieldnames.extend([f"{metric_key}_mean", f"{metric_key}_std"])

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_dir / "innovation2_final_summary.md"
    display_metrics = [
        ("episode_rewards", "Reward"),
        ("episode_completed", "Completed"),
        ("episode_hits", "Hits"),
        ("episode_dropped", "Dropped"),
        ("episode_violation_steps", "Violations"),
        ("episode_avg_csafe", "Safety Cost"),
        ("episode_avg_rperf", "r_perf"),
    ]
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Innovation 2 Final Summary\n\n")
        f.write(f"Tail window: last {tail_n} episodes.\n\n")
        f.write("| Method | Episodes | " + " | ".join(name for _, name in display_metrics) + " |\n")
        f.write("|---|---:|" + "|".join("---:" for _ in display_metrics) + "|\n")
        for row in rows:
            values = [f"{float(row[f'{metric}_mean']):.2f}" for metric, _ in display_metrics]
            f.write(f"| {row['label']} | {row['episodes']} | " + " | ".join(values) + " |\n")

        base, cur = rows
        f.write("\n## Tail-Mean Delta\n\n")
        f.write("| Metric | Curriculum - Safe-C |\n")
        f.write("|---|---:|\n")
        for metric, name in display_metrics:
            delta = float(cur[f"{metric}_mean"]) - float(base[f"{metric}_mean"])
            f.write(f"| {name} | {delta:.2f} |\n")
    return csv_path, md_path


def _svg_escape(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def save_plots(
    rows: List[Dict[str, float | str]],
    all_metrics: Dict[str, Dict[str, np.ndarray]],
    output_dir: Path,
    smooth_window: int,
) -> List[Path]:
    if HAS_MPL:
        return [
            save_reward_curves_png(all_metrics, output_dir / "innovation2_reward_curves.png", smooth_window),
            save_grouped_bar_png(rows, ["episode_completed", "episode_hits", "episode_avg_rperf"], output_dir / "innovation2_performance_metrics.png", "Performance Metrics"),
            save_grouped_bar_png(rows, ["episode_violation_steps", "episode_avg_csafe", "episode_avg_failed_offloads"], output_dir / "innovation2_safety_metrics.png", "Safety Metrics"),
            save_delta_bar_png(rows, output_dir / "innovation2_tail_delta_metrics.png"),
            save_curriculum_diagnostics_png(all_metrics, output_dir / "innovation2_curriculum_diagnostics.png"),
        ]
    return [
        save_reward_curves_svg(all_metrics, output_dir / "innovation2_reward_curves.svg", smooth_window),
        save_grouped_bar_svg(rows, ["episode_completed", "episode_hits", "episode_avg_rperf"], output_dir / "innovation2_performance_metrics.svg", "Performance Metrics"),
        save_grouped_bar_svg(rows, ["episode_violation_steps", "episode_avg_csafe", "episode_avg_failed_offloads"], output_dir / "innovation2_safety_metrics.svg", "Safety Metrics"),
        save_delta_bar_svg(rows, output_dir / "innovation2_tail_delta_metrics.svg"),
        save_curriculum_diagnostics_svg(all_metrics, output_dir / "innovation2_curriculum_diagnostics.svg"),
    ]


def save_reward_curves_png(all_metrics: Dict[str, Dict[str, np.ndarray]], output_path: Path, smooth_window: int) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for scheme_key, label, color in SCHEMES:
        rewards = all_metrics[scheme_key]["episode_rewards"]
        ax.plot(rewards, color=color, linewidth=0.8, alpha=0.25)
        ma_x, ma_y = moving_average(rewards, smooth_window)
        if ma_y is not None:
            ax.plot(ma_x, ma_y, color=color, linewidth=2.0, label=f"{label} MA-{smooth_window}")
        else:
            ax.plot(rewards, color=color, linewidth=1.4, label=label)
    ax.set_title("Reward Curves")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.grid(True, alpha=0.28)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_grouped_bar_png(
    rows: List[Dict[str, float | str]],
    metric_keys: List[str],
    output_path: Path,
    title: str,
) -> Path:
    labels_by_metric = {key: label for key, label, _ in METRICS}
    x = np.arange(len(metric_keys))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for i, (scheme_key, label, color) in enumerate(SCHEMES):
        row = next(row for row in rows if row["scheme"] == scheme_key)
        values = [float(row[f"{metric}_mean"]) for metric in metric_keys]
        bars = ax.bar(x + (i - 0.5) * width, values, width, label=label, color=color, alpha=0.88)
        ax.bar_label(bars, labels=[f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}" for v in values], padding=3, fontsize=8)
    if len(rows) == 2:
        base = rows[0]
        cur = rows[1]
        for xi, metric in enumerate(metric_keys):
            base_v = float(base[f"{metric}_mean"])
            cur_v = float(cur[f"{metric}_mean"])
            delta = cur_v - base_v
            ymax = max(base_v, cur_v)
            ax.annotate(
                f"Δ {delta:+.2f}" if abs(delta) < 10 else f"Δ {delta:+.1f}",
                xy=(xi, ymax),
                xytext=(0, 22),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                fontweight="semibold",
                color="#333",
                arrowprops=dict(arrowstyle="-", color="#666", lw=0.8),
            )
    ax.set_xticks(x)
    ax.set_xticklabels([labels_by_metric[metric] for metric in metric_keys])
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_delta_bar_png(rows: List[Dict[str, float | str]], output_path: Path) -> Path:
    base, cur = rows
    metric_keys = [
        "episode_rewards",
        "episode_completed",
        "episode_hits",
        "episode_dropped",
        "episode_violation_steps",
        "episode_avg_csafe",
        "episode_avg_rperf",
    ]
    labels_by_metric = {key: label for key, label, _ in METRICS}
    deltas = np.asarray([
        float(cur[f"{metric}_mean"]) - float(base[f"{metric}_mean"])
        for metric in metric_keys
    ])
    colors = ["#2ca02c" if value >= 0 else "#d62728" for value in deltas]
    # For metrics where lower is better, invert color semantics.
    lower_better = {"episode_dropped", "episode_violation_steps", "episode_avg_csafe"}
    for i, metric in enumerate(metric_keys):
        if metric in lower_better:
            colors[i] = "#2ca02c" if deltas[i] <= 0 else "#d62728"

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(metric_keys))
    bars = ax.bar(x, deltas, color=colors, alpha=0.88)
    ax.axhline(0, color="#333", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([labels_by_metric[metric] for metric in metric_keys], rotation=20, ha="right")
    ax.set_ylabel("Curriculum-Safe minus Safe-C")
    ax.set_title("Tail-Mean Delta Metrics")
    ax.grid(True, axis="y", alpha=0.28)
    ax.bar_label(bars, labels=[f"{v:+.2f}" if abs(v) < 10 else f"{v:+.1f}" for v in deltas], padding=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_curriculum_diagnostics_png(all_metrics: Dict[str, Dict[str, np.ndarray]], output_path: Path) -> Path:
    cur = all_metrics["curriculum_safe_target_final"]
    episodes = np.arange(1, len(cur["episode_curriculum_level"]) + 1)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.4), sharex=True)

    levels = cur["episode_curriculum_level"]
    axes[0].step(episodes, levels, where="post", color="#9467bd", linewidth=1.8)
    max_level = int(np.nanmax(levels)) if levels.size else 0
    max_level = min(max_level, len(CURRICULUM_LEVEL_NAMES) - 1)
    ticks = np.arange(max_level + 1)
    axes[0].set_yticks(ticks)
    axes[0].set_yticklabels(CURRICULUM_LEVEL_NAMES[: max_level + 1])
    axes[0].set_ylim(-0.2, max_level + 0.4)
    axes[0].set_ylabel("Difficulty")
    axes[0].set_title("Curriculum Difficulty")

    axes[1].plot(cur["episode_value_loss_ema"], color="#1f77b4", linewidth=1.5)
    axes[1].set_ylabel("Value Loss EMA")
    axes[1].set_title("Critic Value-Loss EMA")

    axes[2].plot(cur["episode_learning_progress"], color="#2ca02c", linewidth=1.5)
    axes[2].set_ylabel("Learning Progress")
    axes[2].set_xlabel("Episode")
    axes[2].set_title("Learning Progress")

    for ax in axes:
        ax.grid(True, alpha=0.28)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _simple_svg_chart(output_path: Path, title: str, lines: List[str]) -> Path:
    body = "\n".join(lines)
    output_path.write_text(
        "\n".join([
            '<svg xmlns="http://www.w3.org/2000/svg" width="980" height="560" viewBox="0 0 980 560">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.grid{stroke:#ddd;stroke-width:1}.axis{stroke:#222;stroke-width:1.2}</style>',
            f'<text x="490" y="30" text-anchor="middle" font-size="18">{_svg_escape(title)}</text>',
            body,
            '</svg>',
        ]),
        encoding="utf-8",
    )
    return output_path


def save_reward_curves_svg(all_metrics: Dict[str, Dict[str, np.ndarray]], output_path: Path, smooth_window: int) -> Path:
    width, height = 980, 560
    left, right, top, bottom = 80, 35, 58, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    series = []
    for scheme_key, label, color in SCHEMES:
        rewards = all_metrics[scheme_key]["episode_rewards"]
        x, y = moving_average(rewards, smooth_window)
        if y is None:
            x, y = np.arange(rewards.size), rewards
        series.append((label, color, x, y))
    all_x = np.concatenate([x for _, _, x, _ in series])
    all_y = np.concatenate([y for _, _, _, y in series])
    x_min, x_max = float(all_x.min()), float(all_x.max())
    y_min, y_max = float(all_y.min()), float(all_y.max())
    y_pad = max((y_max - y_min) * 0.1, 1.0)
    y_min -= y_pad
    y_max += y_pad

    def sx(v: float) -> float:
        return left + (v - x_min) / max(x_max - x_min, 1e-9) * plot_w

    def sy(v: float) -> float:
        return top + plot_h - (v - y_min) / max(y_max - y_min, 1e-9) * plot_h

    lines = []
    for tick in np.linspace(y_min, y_max, 5):
        y = sy(float(tick))
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
        lines.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11">{tick:.1f}</text>')
    lines.extend([
        f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>',
        f'<text x="{left + plot_w / 2}" y="{height - 24}" text-anchor="middle" font-size="13">Episode</text>',
        f'<text x="22" y="{top + plot_h / 2}" transform="rotate(-90 22 {top + plot_h / 2})" text-anchor="middle" font-size="13">Episode Reward</text>',
    ])
    for i, (label, color, xs, ys) in enumerate(series):
        pts = " ".join(f"{sx(float(x)):.1f},{sy(float(y)):.1f}" for x, y in zip(xs, ys))
        lines.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<rect x="{left + 20 + i * 220}" y="{height - 52}" width="14" height="14" fill="{color}"/>')
        lines.append(f'<text x="{left + 40 + i * 220}" y="{height - 40}" font-size="12">{_svg_escape(label)}</text>')
    return _simple_svg_chart(output_path, f"Reward Curves (MA-{smooth_window})", lines)


def save_grouped_bar_svg(rows: List[Dict[str, float | str]], metric_keys: List[str], output_path: Path, title: str) -> Path:
    width, height = 980, 560
    left, right, top, bottom = 80, 35, 58, 105
    plot_w, plot_h = width - left - right, height - top - bottom
    labels_by_metric = {key: label for key, label, _ in METRICS}
    values = np.asarray([[float(row[f"{metric}_mean"]) for metric in metric_keys] for row in rows])
    max_v = max(float(np.nanmax(values)) * 1.15, 1.0)
    group_w = plot_w / len(metric_keys)
    bar_w = min(70, group_w / 3.2)

    def sy(v: float) -> float:
        return top + plot_h - (v / max_v) * plot_h

    lines = []
    for tick in np.linspace(0, max_v, 5):
        y = sy(float(tick))
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
        lines.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11">{tick:.1f}</text>')
    lines.extend([
        f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>',
    ])
    for mi, metric in enumerate(metric_keys):
        center = left + group_w * (mi + 0.5)
        for ri, row in enumerate(rows):
            _, _, color = next(item for item in SCHEMES if item[0] == row["scheme"])
            value = float(row[f"{metric}_mean"])
            x = center + (ri - 0.5) * bar_w
            y = sy(value)
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 6:.1f}" height="{top + plot_h - y:.1f}" fill="{color}" opacity="0.88"/>')
            label = f"{value:.2f}" if abs(value) < 10 else f"{value:.0f}"
            lines.append(f'<text x="{x + (bar_w - 6) / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle" font-size="10">{label}</text>')
        if len(rows) == 2:
            delta = float(rows[1][f"{metric}_mean"]) - float(rows[0][f"{metric}_mean"])
            ymax = max(float(rows[0][f"{metric}_mean"]), float(rows[1][f"{metric}_mean"]))
            dlabel = f"Δ {delta:+.2f}" if abs(delta) < 10 else f"Δ {delta:+.1f}"
            lines.append(
                f'<text x="{center:.1f}" y="{sy(ymax) - 28:.1f}" text-anchor="middle" '
                f'font-size="12" font-weight="600">{_svg_escape(dlabel)}</text>'
            )
        lines.append(f'<text x="{center:.1f}" y="{top + plot_h + 26}" text-anchor="middle" font-size="12">{_svg_escape(labels_by_metric[metric])}</text>')
    for i, (_, label, color) in enumerate(SCHEMES):
        x = left + 220 + i * 230
        lines.append(f'<rect x="{x}" y="{height - 48}" width="14" height="14" fill="{color}"/>')
        lines.append(f'<text x="{x + 20}" y="{height - 36}" font-size="12">{_svg_escape(label)}</text>')
    return _simple_svg_chart(output_path, title, lines)


def save_delta_bar_svg(rows: List[Dict[str, float | str]], output_path: Path) -> Path:
    base, cur = rows
    metric_keys = [
        "episode_rewards",
        "episode_completed",
        "episode_hits",
        "episode_dropped",
        "episode_violation_steps",
        "episode_avg_csafe",
        "episode_avg_rperf",
    ]
    labels_by_metric = {key: label for key, label, _ in METRICS}
    deltas = np.asarray([
        float(cur[f"{metric}_mean"]) - float(base[f"{metric}_mean"])
        for metric in metric_keys
    ])
    width, height = 1020, 560
    left, right, top, bottom = 80, 35, 58, 125
    plot_w, plot_h = width - left - right, height - top - bottom
    max_abs = max(float(np.nanmax(np.abs(deltas))) * 1.2, 1.0)
    group_w = plot_w / len(metric_keys)
    bar_w = min(70, group_w * 0.58)

    def sy(v: float) -> float:
        return top + plot_h / 2 - (v / max_abs) * (plot_h / 2)

    zero_y = sy(0.0)
    lower_better = {"episode_dropped", "episode_violation_steps", "episode_avg_csafe"}
    lines = [
        f'<line class="axis" x1="{left}" y1="{zero_y:.1f}" x2="{left + plot_w}" y2="{zero_y:.1f}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>',
        f'<text x="24" y="{top + plot_h / 2}" transform="rotate(-90 24 {top + plot_h / 2})" text-anchor="middle" font-size="12">Curriculum-Safe minus Safe-C</text>',
    ]
    for tick in np.linspace(-max_abs, max_abs, 5):
        y = sy(float(tick))
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
        lines.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11">{tick:.1f}</text>')
    for i, (metric, delta) in enumerate(zip(metric_keys, deltas)):
        center = left + group_w * (i + 0.5)
        y = sy(float(delta))
        h = abs(zero_y - y)
        y_rect = min(y, zero_y)
        beneficial = delta <= 0 if metric in lower_better else delta >= 0
        color = "#2ca02c" if beneficial else "#d62728"
        lines.append(f'<rect x="{center - bar_w / 2:.1f}" y="{y_rect:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}" opacity="0.88"/>')
        label = f"{delta:+.2f}" if abs(delta) < 10 else f"{delta:+.1f}"
        text_y = y - 8 if delta >= 0 else y + 18
        lines.append(f'<text x="{center:.1f}" y="{text_y:.1f}" text-anchor="middle" font-size="11" font-weight="600">{label}</text>')
        lines.append(f'<text x="{center:.1f}" y="{top + plot_h + 24}" text-anchor="middle" font-size="11" transform="rotate(20 {center:.1f} {top + plot_h + 24})">{_svg_escape(labels_by_metric[metric])}</text>')
    return _simple_svg_chart(output_path, "Tail-Mean Delta Metrics", lines)


def save_curriculum_diagnostics_svg(all_metrics: Dict[str, Dict[str, np.ndarray]], output_path: Path) -> Path:
    cur = all_metrics["curriculum_safe_target_final"]
    width, height = 980, 680
    left, right, top, bottom = 110, 35, 58, 55
    panel_h = 170
    plot_w = width - left - right
    lines = []
    panels = [
        ("Curriculum Difficulty", cur["episode_curriculum_level"], "Difficulty", "#9467bd"),
        ("Critic Value-Loss EMA", cur["episode_value_loss_ema"], "EMA", "#1f77b4"),
        ("Learning Progress", cur["episode_learning_progress"], "Progress", "#2ca02c"),
    ]
    for pi, (panel_title, values, ylabel, color) in enumerate(panels):
        ptop = top + pi * (panel_h + 28)
        values = np.asarray(values, dtype=np.float64)
        x_min, x_max = 0.0, float(values.size - 1)
        if pi == 0:
            y_min, y_max = -0.2, min(float(np.nanmax(values)) + 0.4, len(CURRICULUM_LEVEL_NAMES) - 0.6)
        else:
            finite_values = values[np.isfinite(values)]
            if finite_values.size == 0:
                finite_values = np.asarray([0.0], dtype=np.float64)
            y_min, y_max = float(np.nanmin(finite_values)), float(np.nanmax(finite_values))
            pad = max((y_max - y_min) * 0.12, 1e-6)
            y_min -= pad
            y_max += pad

        def sx(v: float) -> float:
            return left + (v - x_min) / max(x_max - x_min, 1e-9) * plot_w

        def sy(v: float) -> float:
            return ptop + panel_h - (v - y_min) / max(y_max - y_min, 1e-9) * panel_h

        lines.append(f'<text x="{left}" y="{ptop - 8}" font-size="14" font-weight="600">{_svg_escape(panel_title)}</text>')
        lines.append(f'<line class="axis" x1="{left}" y1="{ptop + panel_h}" x2="{left + plot_w}" y2="{ptop + panel_h}"/>')
        lines.append(f'<line class="axis" x1="{left}" y1="{ptop}" x2="{left}" y2="{ptop + panel_h}"/>')
        if pi == 0:
            max_level = int(np.nanmax(values)) if values.size else 0
            for level in range(max_level + 1):
                y = sy(float(level))
                lines.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
                lines.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11">{CURRICULUM_LEVEL_NAMES[level]}</text>')
            pts = " ".join(f"{sx(float(i)):.1f},{sy(float(v)):.1f}" for i, v in enumerate(values))
        else:
            for tick in np.linspace(y_min, y_max, 4):
                y = sy(float(tick))
                lines.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
                lines.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11">{tick:.2f}</text>')
            pts = " ".join(
                f"{sx(float(i)):.1f},{sy(float(v)):.1f}"
                for i, v in enumerate(values)
                if np.isfinite(v)
            )
        if pts:
            lines.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<text x="24" y="{ptop + panel_h / 2}" transform="rotate(-90 24 {ptop + panel_h / 2})" text-anchor="middle" font-size="12">{_svg_escape(ylabel)}</text>')
    lines.append(f'<text x="{left + plot_w / 2}" y="{height - 18}" text-anchor="middle" font-size="13">Episode</text>')
    return _simple_svg_chart(output_path, "Curriculum Diagnostics", lines)


def main() -> None:
    args = parse_args()
    artifact_root = Path(args.artifact_root)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = artifact_root / "generated_results" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, all_metrics = load_all(artifact_root, args.tail_n)
    csv_path, md_path = write_summary(rows, output_dir, args.tail_n)
    figure_paths = save_plots(rows, all_metrics, output_dir, args.smooth_window)

    print(f"Saved Innovation 2 final results to: {output_dir}")
    print(f"Summary CSV: {csv_path}")
    print(f"Summary Markdown: {md_path}")
    for path in figure_paths:
        print(f"Figure: {path}")

    print("\nTail means:")
    for row in rows:
        print(
            f"  {row['label']:<22} "
            f"reward={float(row['episode_rewards_mean']):.2f}, "
            f"completed={float(row['episode_completed_mean']):.2f}, "
            f"hits={float(row['episode_hits_mean']):.2f}, "
            f"violations={float(row['episode_violation_steps_mean']):.2f}, "
            f"csafe={float(row['episode_avg_csafe_mean']):.2f}"
        )


if __name__ == "__main__":
    main()
