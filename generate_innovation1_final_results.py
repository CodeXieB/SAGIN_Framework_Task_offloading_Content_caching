"""
Generate final tables and figures for Innovation 1 experiments.

Outputs are saved under:
    artifacts/generated_results/<timestamp>/

The script reads only existing metrics.npz files and does not change training
code or experiment artifacts.

Example:
    python generate_innovation1_final_results.py
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
    ("baseline", "Baseline PPO", "#1f77b4"),
    ("safe_a", "Safe-A Hard", "#d62728"),
    ("safe_b", "Safe-B Soft", "#2ca02c"),
    ("safe_c", "Safe-C Balanced", "#ff7f0e"),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fixed final tables and plots for Innovation 1."
    )
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--tail-n", type=int, default=50)
    parser.add_argument("--run-id", default=None, help="Optional fixed output directory name.")
    return parser.parse_args()


def tail_stats(values: np.ndarray, tail_n: int) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    tail = values[-tail_n:] if values.size >= tail_n else values
    return float(np.nanmean(tail)), float(np.nanstd(tail))


def load_summary(artifact_root: Path, tail_n: int) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    for scheme_key, scheme_label, _ in SCHEMES:
        metrics_path = artifact_root / scheme_key / "metrics.npz"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
        data = np.load(metrics_path, allow_pickle=False)
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
    return rows


def write_csv(rows: List[Dict[str, float | str]], output_dir: Path, tail_n: int) -> Path:
    path = output_dir / "innovation1_final_summary.csv"
    fieldnames = ["scheme", "label", "episodes"]
    for metric_key, _, _ in METRICS:
        fieldnames.extend([f"{metric_key}_mean", f"{metric_key}_std"])

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_dir / "innovation1_final_summary.md"
    display_metrics = [
        ("episode_completed", "Completed"),
        ("episode_hits", "Hits"),
        ("episode_dropped", "Dropped"),
        ("episode_violation_steps", "Violations"),
        ("episode_avg_csafe", "Safety Cost"),
        ("episode_rewards", "Reward"),
    ]
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# Innovation 1 Final Summary\n\n")
        f.write(f"Tail window: last {tail_n} episodes.\n\n")
        f.write("| Method | " + " | ".join(name for _, name in display_metrics) + " |\n")
        f.write("|---|" + "|".join("---:" for _ in display_metrics) + "|\n")
        for row in rows:
            values = []
            for metric_key, _ in display_metrics:
                values.append(f"{float(row[f'{metric_key}_mean']):.2f}")
            f.write(f"| {row['label']} | " + " | ".join(values) + " |\n")
    return path


def _svg_escape(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _save_grouped_bar_svg(
    rows: List[Dict[str, float | str]],
    metric_keys: Iterable[str],
    output_path: Path,
    title: str,
) -> Path:
    metric_keys = list(metric_keys)
    svg_path = output_path.with_suffix(".svg")
    width, height = 980, 560
    left, right, top, bottom = 80, 30, 58, 120
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = np.asarray(
        [[float(row[f"{metric}_mean"]) for metric in metric_keys] for row in rows],
        dtype=np.float64,
    )
    max_v = max(float(np.nanmax(values)) * 1.12, 1.0)
    group_w = plot_w / len(metric_keys)
    bar_w = min(32, group_w / (len(rows) + 1.2))
    labels_by_metric = {key: label for key, label, _ in METRICS}
    colors_by_scheme = {key: color for key, _, color in SCHEMES}

    def sy(v: float) -> float:
        return top + plot_h - (v / max_v) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.grid{stroke:#ddd;stroke-width:1}.axis{stroke:#222;stroke-width:1.3}</style>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-size="18">{_svg_escape(title)}</text>',
    ]
    for tick in np.linspace(0, max_v, 5):
        y = sy(float(tick))
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
        lines.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11">{tick:.1f}</text>')
    lines.extend([
        f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>',
    ])

    for mi, metric in enumerate(metric_keys):
        group_center = left + group_w * (mi + 0.5)
        for ri, row in enumerate(rows):
            scheme_key = str(row["scheme"])
            value = float(row[f"{metric}_mean"])
            x = group_center - (len(rows) * bar_w) / 2 + ri * bar_w
            y = sy(value)
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 3:.1f}" '
                f'height="{top + plot_h - y:.1f}" fill="{colors_by_scheme[scheme_key]}" opacity="0.88"/>'
            )
        lines.append(
            f'<text x="{group_center:.1f}" y="{top + plot_h + 24}" text-anchor="middle" '
            f'font-size="12">{_svg_escape(labels_by_metric[metric])}</text>'
        )

    legend_x = left + 20
    legend_y = height - 62
    for i, (scheme_key, label, color) in enumerate(SCHEMES):
        x = legend_x + i * 210
        lines.append(f'<rect x="{x}" y="{legend_y}" width="14" height="14" fill="{color}"/>')
        lines.append(f'<text x="{x + 20}" y="{legend_y + 12}" font-size="12">{_svg_escape(label)}</text>')
    lines.append("</svg>")
    svg_path.write_text("\n".join(lines), encoding="utf-8")
    return svg_path


def _save_tradeoff_svg(rows: List[Dict[str, float | str]], output_path: Path) -> Path:
    svg_path = output_path.with_suffix(".svg")
    width, height = 920, 580
    left, right, top, bottom = 95, 35, 60, 85
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = np.asarray([float(row["episode_violation_steps_mean"]) for row in rows])
    ys = np.asarray([float(row["episode_completed_mean"]) for row in rows])
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    x_pad = max((x_max - x_min) * 0.14, 1.0)
    y_pad = max((y_max - y_min) * 0.14, 1.0)
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    def sx(v: float) -> float:
        return left + (v - x_min) / max(x_max - x_min, 1e-9) * plot_w

    def sy(v: float) -> float:
        return top + plot_h - (v - y_min) / max(y_max - y_min, 1e-9) * plot_h

    colors = {key: color for key, _, color in SCHEMES}
    offsets = {
        "baseline": (12, 8, "start"),
        "safe_a": (-52, -22, "end"),
        "safe_b": (28, 30, "start"),
        "safe_c": (12, -18, "start"),
    }
    tags = {"baseline": "B", "safe_a": "A", "safe_b": "B", "safe_c": "C"}

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.grid{stroke:#ddd;stroke-width:1}.axis{stroke:#222;stroke-width:1.3}</style>',
        '<text x="460" y="30" text-anchor="middle" font-size="18">Safety-Performance Tradeoff</text>',
    ]
    for tick in np.linspace(x_min, x_max, 5):
        x = sx(float(tick))
        lines.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}"/>')
        lines.append(f'<text x="{x:.1f}" y="{top + plot_h + 24}" text-anchor="middle" font-size="11">{tick:.1f}</text>')
    for tick in np.linspace(y_min, y_max, 5):
        y = sy(float(tick))
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
        lines.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11">{tick:.1f}</text>')
    lines.extend([
        f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>',
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 26}" text-anchor="middle" font-size="13">Violation Steps (lower is better)</text>',
        f'<text x="22" y="{top + plot_h / 2:.1f}" transform="rotate(-90 22 {top + plot_h / 2:.1f})" text-anchor="middle" font-size="13">Completed Tasks (higher is better)</text>',
        f'<text x="{left + 12}" y="{top + 18}" font-size="12" fill="#2ca02c">better</text>',
    ])

    for row in rows:
        key = str(row["scheme"])
        label = str(row["label"])
        x = sx(float(row["episode_violation_steps_mean"]))
        y = sy(float(row["episode_completed_mean"]))
        color = colors[key]
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" fill="{color}" stroke="#111"/>')
        lines.append(
            f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" '
            f'font-size="10" font-weight="700" fill="white">{tags[key]}</text>'
        )
        dx, dy, anchor = offsets[key]
        lx, ly = x + dx, y + dy
        if key in {"safe_a", "safe_b"}:
            lines.append(
                f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{lx:.1f}" y2="{ly:.1f}" '
                f'stroke="{color}" stroke-width="1" stroke-dasharray="3,3"/>'
            )
        lines.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" '
            f'font-size="12" font-weight="600">{_svg_escape(label)}</text>'
        )

    lines.append("</svg>")
    svg_path.write_text("\n".join(lines), encoding="utf-8")
    return svg_path


def save_plots(rows: List[Dict[str, float | str]], output_dir: Path) -> List[Path]:
    paths: List[Path] = []
    performance_metrics = ["episode_completed", "episode_hits", "episode_rewards"]
    safety_metrics = ["episode_violation_steps", "episode_avg_csafe", "episode_avg_failed_offloads"]

    if HAS_MPL:
        paths.append(save_tradeoff_png(rows, output_dir / "innovation1_safety_performance_tradeoff.png"))
        paths.append(save_grouped_bar_png(rows, performance_metrics, output_dir / "innovation1_performance_metrics.png", "Performance Metrics"))
        paths.append(save_grouped_bar_png(rows, safety_metrics, output_dir / "innovation1_safety_metrics.png", "Safety Metrics"))
    else:
        paths.append(_save_tradeoff_svg(rows, output_dir / "innovation1_safety_performance_tradeoff.png"))
        paths.append(_save_grouped_bar_svg(rows, performance_metrics, output_dir / "innovation1_performance_metrics.png", "Performance Metrics"))
        paths.append(_save_grouped_bar_svg(rows, safety_metrics, output_dir / "innovation1_safety_metrics.png", "Safety Metrics"))
    return paths


def save_tradeoff_png(rows: List[Dict[str, float | str]], output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    offsets = {
        "baseline": (10, 8, "left"),
        "safe_a": (-58, -22, "right"),
        "safe_b": (28, 30, "left"),
        "safe_c": (10, -18, "left"),
    }
    for scheme_key, label, color in SCHEMES:
        row = next(row for row in rows if row["scheme"] == scheme_key)
        x = float(row["episode_violation_steps_mean"])
        y = float(row["episode_completed_mean"])
        ax.scatter(x, y, s=150, color=color, edgecolor="black", linewidth=0.8, zorder=3, label=label)
        ax.text(x, y, {"baseline": "B", "safe_a": "A", "safe_b": "B", "safe_c": "C"}[scheme_key],
                ha="center", va="center", color="white", fontsize=8, fontweight="bold", zorder=4)
        dx, dy, align = offsets[scheme_key]
        ax.annotate(
            label,
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9,
            fontweight="semibold",
            ha="right" if align == "right" else "left",
            arrowprops=dict(arrowstyle="-", color=color, lw=0.9, linestyle="--", alpha=0.8)
            if scheme_key in {"safe_a", "safe_b"}
            else None,
        )
    ax.set_xlabel("Violation Steps (lower is better)")
    ax.set_ylabel("Completed Tasks (higher is better)")
    ax.set_title("Safety-Performance Tradeoff")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="best", frameon=True)
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
    width = 0.18
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for i, (scheme_key, label, color) in enumerate(SCHEMES):
        row = next(row for row in rows if row["scheme"] == scheme_key)
        values = [float(row[f"{metric}_mean"]) for metric in metric_keys]
        ax.bar(x + (i - 1.5) * width, values, width, label=label, color=color, alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels([labels_by_metric[metric] for metric in metric_keys])
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    artifact_root = Path(args.artifact_root)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = artifact_root / "generated_results" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_summary(artifact_root, args.tail_n)
    csv_path = write_csv(rows, output_dir, args.tail_n)
    plot_paths = save_plots(rows, output_dir)

    print(f"Saved Innovation 1 final results to: {output_dir}")
    print(f"Summary CSV: {csv_path}")
    for path in plot_paths:
        print(f"Figure: {path}")
    print("\nTail means:")
    for row in rows:
        print(
            f"  {row['label']:<18} "
            f"completed={float(row['episode_completed_mean']):.2f}, "
            f"violations={float(row['episode_violation_steps_mean']):.2f}, "
            f"csafe={float(row['episode_avg_csafe_mean']):.2f}, "
            f"reward={float(row['episode_rewards_mean']):.2f}"
        )


if __name__ == "__main__":
    main()
