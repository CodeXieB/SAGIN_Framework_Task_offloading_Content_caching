"""
Generate all-metric training-curve dashboards for Innovation 1 and Innovation 2.

Each innovation gets two integrated figures:
    Innovation 1:
        1) reward/performance curves
        2) safety/optimization curves
    Innovation 2:
        1) reward/performance curves
        2) safety/curriculum/optimization curves

Each subplot overlays all relevant models for that metric.

Outputs are saved under:
    artifacts/generated_results/<timestamp>/all_metric_curves/
"""
from __future__ import annotations

import argparse
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


INNOVATION1_SCHEMES: List[Tuple[str, str, str]] = [
    ("baseline", "Baseline", "#1f77b4"),
    ("safe_a", "Safe-A", "#d62728"),
    ("safe_b", "Safe-B", "#2ca02c"),
    ("safe_c", "Safe-C", "#ff7f0e"),
]

INNOVATION2_SCHEMES: List[Tuple[str, str, str]] = [
    ("safe_c_800", "Safe-C-800", "#ff7f0e"),
    ("curriculum_safe_target_final", "Curriculum-Safe-800", "#9467bd"),
]

CURRICULUM_LEVEL_NAMES = ["Easy", "Medium", "Target", "Hard"]
TARGET_LEVEL = 2

METRIC_LABELS = {
    "episode_rewards": "Episode Reward",
    "episode_completed": "Completed Tasks",
    "episode_hits": "Cache Hits",
    "episode_dropped": "Dropped Tasks",
    "episode_avg_rperf": "Average Performance Reward",
    "episode_violation_steps": "Violation Steps",
    "episode_avg_csafe": "Average Safety Cost",
    "episode_avg_failed_offloads": "Average Failed Offloads",
    "episode_avg_energy_ratio": "Average Energy Ratio",
    "episode_avg_queue_ratio": "Average Queue Ratio",
    "episode_v_loss": "Value Loss",
    "episode_entropy": "Policy Entropy",
    "episode_curriculum_level": "Curriculum Difficulty",
    "episode_value_loss_ema": "Value Loss EMA",
    "episode_learning_progress": "Learning Progress",
}

INNOVATION1_PERFORMANCE = [
    "episode_rewards",
    "episode_completed",
    "episode_hits",
    "episode_dropped",
    "episode_avg_rperf",
]

INNOVATION1_SAFETY_OPT = [
    "episode_violation_steps",
    "episode_avg_csafe",
    "episode_avg_failed_offloads",
    "episode_avg_energy_ratio",
    "episode_avg_queue_ratio",
    "episode_v_loss",
    "episode_entropy",
]

INNOVATION2_PERFORMANCE = [
    "episode_rewards",
    "episode_completed",
    "episode_hits",
    "episode_dropped",
    "episode_avg_rperf",
]

INNOVATION2_SAFETY_CURRICULUM_OPT = [
    "episode_violation_steps",
    "episode_avg_csafe",
    "episode_avg_failed_offloads",
    "episode_curriculum_level",
    "episode_value_loss_ema",
    "episode_learning_progress",
    "episode_v_loss",
    "episode_entropy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate all metric curve dashboards.")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--smooth-window", type=int, default=20)
    return parser.parse_args()


def load_metrics(
    artifact_root: Path,
    schemes: List[Tuple[str, str, str]],
) -> Dict[str, Dict[str, np.ndarray]]:
    result: Dict[str, Dict[str, np.ndarray]] = {}
    for key, _, _ in schemes:
        metrics_path = artifact_root / key / "metrics.npz"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
        data = np.load(metrics_path, allow_pickle=False)
        result[key] = {name: data[name] for name in data.files}
    return result


def moving_average(values: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    finite_mask = np.isfinite(values)
    if not finite_mask.all():
        values = values.copy()
        finite_values = values[finite_mask]
        fill = float(np.nanmean(finite_values)) if finite_values.size else 0.0
        values[~finite_mask] = fill
    if window <= 1 or values.size < window:
        return np.arange(values.size), values
    kernel = np.ones(window, dtype=np.float64) / float(window)
    smoothed = np.convolve(values, kernel, mode="valid")
    x = np.arange(window - 1, window - 1 + smoothed.size)
    return x, smoothed


def is_curriculum_scheme(scheme_key: str) -> bool:
    return "curriculum" in scheme_key


def plotted_curriculum_levels(scheme_key: str, values: np.ndarray) -> Tuple[np.ndarray, str]:
    values = np.asarray(values, dtype=np.float64)
    if is_curriculum_scheme(scheme_key):
        return values, ""
    # Non-curriculum experiments run directly on the target environment. Their
    # saved level value is only a placeholder, so plot it as fixed Target.
    return np.full(values.shape, TARGET_LEVEL, dtype=np.float64), " (Fixed Target)"


def _svg_escape(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def save_dashboard(
    output_path: Path,
    title: str,
    schemes: List[Tuple[str, str, str]],
    metrics_by_scheme: Dict[str, Dict[str, np.ndarray]],
    metric_keys: List[str],
    smooth_window: int,
) -> Path:
    if HAS_MPL:
        return save_dashboard_png(output_path.with_suffix(".png"), title, schemes, metrics_by_scheme, metric_keys, smooth_window)
    return save_dashboard_svg(output_path.with_suffix(".svg"), title, schemes, metrics_by_scheme, metric_keys, smooth_window)


def save_dashboard_png(
    output_path: Path,
    title: str,
    schemes: List[Tuple[str, str, str]],
    metrics_by_scheme: Dict[str, Dict[str, np.ndarray]],
    metric_keys: List[str],
    smooth_window: int,
) -> Path:
    n = len(metric_keys)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13.5, 3.5 * rows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, metric_key in zip(axes_flat, metric_keys):
        for scheme_key, label, color in schemes:
            data = metrics_by_scheme[scheme_key]
            if metric_key not in data:
                continue
            y = np.asarray(data[metric_key], dtype=np.float64)
            if metric_key == "episode_curriculum_level":
                y, label_suffix = plotted_curriculum_levels(scheme_key, y)
                x = np.arange(y.size)
                ax.step(x, y, where="post", color=color, linewidth=1.8, label=label + label_suffix)
            else:
                x, smoothed = moving_average(y, smooth_window)
                ax.plot(x, smoothed, color=color, linewidth=1.7, label=label)
        ax.set_title(METRIC_LABELS.get(metric_key, metric_key))
        ax.set_xlabel("Episode")
        ax.grid(True, alpha=0.28)
        if metric_key == "episode_curriculum_level":
            max_level = 0
            for scheme_key, _, _ in schemes:
                data = metrics_by_scheme[scheme_key]
                if metric_key in data:
                    vals = np.asarray(data[metric_key], dtype=np.float64)
                    vals, _ = plotted_curriculum_levels(scheme_key, vals)
                    finite = vals[np.isfinite(vals)]
                    if finite.size:
                        max_level = max(max_level, int(np.nanmax(finite)))
            max_level = min(max_level, len(CURRICULUM_LEVEL_NAMES) - 1)
            ticks = np.arange(max_level + 1)
            ax.set_yticks(ticks)
            ax.set_yticklabels(CURRICULUM_LEVEL_NAMES[: max_level + 1])
            ax.set_ylim(-0.2, max_level + 0.4)

    for ax in axes_flat[n:]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(title, y=1.02, fontsize=15, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_dashboard_svg(
    output_path: Path,
    title: str,
    schemes: List[Tuple[str, str, str]],
    metrics_by_scheme: Dict[str, Dict[str, np.ndarray]],
    metric_keys: List[str],
    smooth_window: int,
) -> Path:
    cols = 2
    rows = int(np.ceil(len(metric_keys) / cols))
    panel_w, panel_h = 560, 280
    margin_x, margin_y = 72, 65
    gap_x, gap_y = 45, 65
    width = margin_x * 2 + cols * panel_w + (cols - 1) * gap_x
    height = margin_y * 2 + rows * panel_h + (rows - 1) * gap_y + 45

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.grid{stroke:#ddd;stroke-width:1}.axis{stroke:#222;stroke-width:1.2}</style>',
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" font-size="18" font-weight="700">{_svg_escape(title)}</text>',
    ]

    legend_x = margin_x
    for i, (_, label, color) in enumerate(schemes):
        x = legend_x + i * 180
        lines.append(f'<rect x="{x}" y="45" width="14" height="14" fill="{color}"/>')
        lines.append(f'<text x="{x + 20}" y="57" font-size="12">{_svg_escape(label)}</text>')

    for idx, metric_key in enumerate(metric_keys):
        col = idx % cols
        row = idx // cols
        left = margin_x + col * (panel_w + gap_x)
        top = margin_y + 45 + row * (panel_h + gap_y)
        plot_w = panel_w
        plot_h = panel_h

        series = []
        for scheme_key, label, color in schemes:
            data = metrics_by_scheme[scheme_key]
            if metric_key not in data:
                continue
            y = np.asarray(data[metric_key], dtype=np.float64)
            if metric_key == "episode_curriculum_level":
                y, label_suffix = plotted_curriculum_levels(scheme_key, y)
                x = np.arange(y.size)
                y_plot = y
            else:
                label_suffix = ""
                x, y_plot = moving_average(y, smooth_window)
            finite = np.isfinite(y_plot)
            if finite.any():
                series.append((label + label_suffix, color, x[finite], y_plot[finite]))
        if not series:
            continue

        all_x = np.concatenate([s[2] for s in series])
        all_y = np.concatenate([s[3] for s in series])
        x_min, x_max = float(all_x.min()), float(all_x.max())
        if metric_key == "episode_curriculum_level":
            y_min = -0.2
            y_max = min(float(np.nanmax(all_y)) + 0.4, len(CURRICULUM_LEVEL_NAMES) - 0.6)
        else:
            y_min, y_max = float(all_y.min()), float(all_y.max())
            pad = max((y_max - y_min) * 0.12, 1e-6)
            y_min -= pad
            y_max += pad

        def sx(v: float) -> float:
            return left + (v - x_min) / max(x_max - x_min, 1e-9) * plot_w

        def sy(v: float) -> float:
            return top + plot_h - (v - y_min) / max(y_max - y_min, 1e-9) * plot_h

        lines.append(f'<text x="{left}" y="{top - 12}" font-size="14" font-weight="600">{_svg_escape(METRIC_LABELS.get(metric_key, metric_key))}</text>')
        lines.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>')
        lines.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')

        if metric_key == "episode_curriculum_level":
            max_level = int(np.nanmax(all_y)) if all_y.size else 0
            max_level = min(max_level, len(CURRICULUM_LEVEL_NAMES) - 1)
            for level in range(max_level + 1):
                y_tick = sy(float(level))
                lines.append(f'<line class="grid" x1="{left}" y1="{y_tick:.1f}" x2="{left + plot_w}" y2="{y_tick:.1f}"/>')
                lines.append(f'<text x="{left - 8}" y="{y_tick + 4:.1f}" text-anchor="end" font-size="10">{CURRICULUM_LEVEL_NAMES[level]}</text>')
        else:
            for tick in np.linspace(y_min, y_max, 4):
                y_tick = sy(float(tick))
                lines.append(f'<line class="grid" x1="{left}" y1="{y_tick:.1f}" x2="{left + plot_w}" y2="{y_tick:.1f}"/>')
                lines.append(f'<text x="{left - 8}" y="{y_tick + 4:.1f}" text-anchor="end" font-size="10">{tick:.2f}</text>')

        for _, color, xs, ys in series:
            points = " ".join(f"{sx(float(x)):.1f},{sy(float(y)):.1f}" for x, y in zip(xs, ys))
            lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<text x="{left + plot_w / 2:.1f}" y="{top + plot_h + 30}" text-anchor="middle" font-size="11">Episode</text>')

    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    artifact_root = Path(args.artifact_root)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = artifact_root / "generated_results" / run_id / "all_metric_curves"
    output_dir.mkdir(parents=True, exist_ok=True)

    innovation1_metrics = load_metrics(artifact_root, INNOVATION1_SCHEMES)
    innovation2_metrics = load_metrics(artifact_root, INNOVATION2_SCHEMES)

    paths = [
        save_dashboard(
            output_dir / "innovation1_performance_curves",
            "Innovation 1 - Reward and Performance Curves",
            INNOVATION1_SCHEMES,
            innovation1_metrics,
            INNOVATION1_PERFORMANCE,
            args.smooth_window,
        ),
        save_dashboard(
            output_dir / "innovation1_safety_optimization_curves",
            "Innovation 1 - Safety and Optimization Curves",
            INNOVATION1_SCHEMES,
            innovation1_metrics,
            INNOVATION1_SAFETY_OPT,
            args.smooth_window,
        ),
        save_dashboard(
            output_dir / "innovation2_performance_curves",
            "Innovation 2 - Reward and Performance Curves",
            INNOVATION2_SCHEMES,
            innovation2_metrics,
            INNOVATION2_PERFORMANCE,
            args.smooth_window,
        ),
        save_dashboard(
            output_dir / "innovation2_safety_curriculum_optimization_curves",
            "Innovation 2 - Safety, Curriculum, and Optimization Curves",
            INNOVATION2_SCHEMES,
            innovation2_metrics,
            INNOVATION2_SAFETY_CURRICULUM_OPT,
            args.smooth_window,
        ),
    ]

    print(f"Saved all metric curves to: {output_dir}")
    for path in paths:
        print(f"Figure: {path}")


if __name__ == "__main__":
    main()
