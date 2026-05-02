"""
Generate fixed-cache ablation results.

Main comparison:
    artifacts/safe_c_800              -> Joint Cache
    artifacts/fixed_cache_safe_c_800  -> Fixed Cache

Outputs are saved under:
    artifacts/generated_results/<timestamp>/
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
    ("safe_c_800", "Joint Cache", "#1f77b4"),
    ("fixed_cache_safe_c_800", "Fixed Cache", "#ff7f0e"),
]

METRICS: List[Tuple[str, str]] = [
    ("episode_rewards", "Reward"),
    ("episode_completed", "Completed"),
    ("episode_hits", "Hits"),
    ("episode_dropped", "Dropped"),
    ("episode_violation_steps", "Violations"),
    ("episode_avg_csafe", "Safety Cost"),
    ("episode_avg_rperf", "r_perf"),
    ("episode_avg_failed_offloads", "Failed Offloads"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fixed-cache comparison results.")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--tail-n", type=int, default=50)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def tail_stats(values: np.ndarray, tail_n: int) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    tail = values[-tail_n:] if values.size >= tail_n else values
    return float(np.nanmean(tail)), float(np.nanstd(tail))


def load_rows(artifact_root: Path, tail_n: int) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    for scheme_key, label, _ in SCHEMES:
        metrics_path = artifact_root / scheme_key / "metrics.npz"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
        data = np.load(metrics_path, allow_pickle=False)
        row: Dict[str, float | str] = {
            "scheme": scheme_key,
            "label": label,
            "episodes": int(len(data["episode_rewards"])),
        }
        for metric_key, _ in METRICS:
            mean, std = tail_stats(data[metric_key], tail_n)
            row[f"{metric_key}_mean"] = mean
            row[f"{metric_key}_std"] = std
        rows.append(row)
    return rows


def write_summary(rows: List[Dict[str, float | str]], output_dir: Path, tail_n: int) -> Tuple[Path, Path]:
    csv_path = output_dir / "fixed_cache_comparison_summary.csv"
    fieldnames = ["scheme", "label", "episodes"]
    for metric_key, _ in METRICS:
        fieldnames.extend([f"{metric_key}_mean", f"{metric_key}_std"])
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_dir / "fixed_cache_comparison_summary.md"
    display = [
        ("episode_rewards", "Reward"),
        ("episode_completed", "Completed"),
        ("episode_hits", "Hits"),
        ("episode_dropped", "Dropped"),
        ("episode_violation_steps", "Violations"),
        ("episode_avg_csafe", "Safety Cost"),
        ("episode_avg_rperf", "r_perf"),
    ]
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Fixed Cache Comparison\n\n")
        f.write(f"Tail window: last {tail_n} episodes.\n\n")
        f.write("| Method | Episodes | " + " | ".join(name for _, name in display) + " |\n")
        f.write("|---|---:|" + "|".join("---:" for _ in display) + "|\n")
        for row in rows:
            values = [f"{float(row[f'{metric}_mean']):.2f}" for metric, _ in display]
            f.write(f"| {row['label']} | {row['episodes']} | " + " | ".join(values) + " |\n")

        joint, fixed = rows
        f.write("\n## Tail-Mean Delta\n\n")
        f.write("| Metric | Joint Cache - Fixed Cache |\n")
        f.write("|---|---:|\n")
        for metric, name in display:
            delta = float(joint[f"{metric}_mean"]) - float(fixed[f"{metric}_mean"])
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


def save_plot(rows: List[Dict[str, float | str]], output_dir: Path) -> Path:
    metric_keys = [
        "episode_completed",
        "episode_hits",
        "episode_avg_rperf",
        "episode_violation_steps",
        "episode_avg_csafe",
    ]
    if HAS_MPL:
        return save_plot_png(rows, metric_keys, output_dir / "fixed_cache_comparison_metrics.png")
    return save_plot_svg(rows, metric_keys, output_dir / "fixed_cache_comparison_metrics.svg")


def save_plot_png(rows: List[Dict[str, float | str]], metric_keys: List[str], output_path: Path) -> Path:
    labels_by_metric = {key: label for key, label in METRICS}
    x = np.arange(len(metric_keys))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for i, (scheme_key, label, color) in enumerate(SCHEMES):
        row = next(row for row in rows if row["scheme"] == scheme_key)
        values = [float(row[f"{metric}_mean"]) for metric in metric_keys]
        bars = ax.bar(x + (i - 0.5) * width, values, width, label=label, color=color, alpha=0.88)
        ax.bar_label(bars, labels=[f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}" for v in values], padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([labels_by_metric[metric] for metric in metric_keys])
    ax.set_title("Fixed Cache Ablation")
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_plot_svg(rows: List[Dict[str, float | str]], metric_keys: List[str], output_path: Path) -> Path:
    width, height = 980, 560
    left, right, top, bottom = 80, 35, 58, 105
    plot_w, plot_h = width - left - right, height - top - bottom
    labels_by_metric = {key: label for key, label in METRICS}
    values = np.asarray([[float(row[f"{metric}_mean"]) for metric in metric_keys] for row in rows])
    max_v = max(float(np.nanmax(values)) * 1.15, 1.0)
    group_w = plot_w / len(metric_keys)
    bar_w = min(60, group_w / 3.2)

    def sy(v: float) -> float:
        return top + plot_h - (v / max_v) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.grid{stroke:#ddd;stroke-width:1}.axis{stroke:#222;stroke-width:1.2}</style>',
        '<text x="490" y="30" text-anchor="middle" font-size="18">Fixed Cache Ablation</text>',
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
        center = left + group_w * (mi + 0.5)
        for ri, row in enumerate(rows):
            _, _, color = next(item for item in SCHEMES if item[0] == row["scheme"])
            value = float(row[f"{metric}_mean"])
            x = center + (ri - 0.5) * bar_w
            y = sy(value)
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 6:.1f}" height="{top + plot_h - y:.1f}" fill="{color}" opacity="0.88"/>')
            label = f"{value:.2f}" if abs(value) < 10 else f"{value:.0f}"
            lines.append(f'<text x="{x + (bar_w - 6) / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle" font-size="10">{label}</text>')
        lines.append(f'<text x="{center:.1f}" y="{top + plot_h + 26}" text-anchor="middle" font-size="12">{_svg_escape(labels_by_metric[metric])}</text>')
    for i, (_, label, color) in enumerate(SCHEMES):
        x = left + 260 + i * 220
        lines.append(f'<rect x="{x}" y="{height - 48}" width="14" height="14" fill="{color}"/>')
        lines.append(f'<text x="{x + 20}" y="{height - 36}" font-size="12">{_svg_escape(label)}</text>')
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    artifact_root = Path(args.artifact_root)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = artifact_root / "generated_results" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(artifact_root, args.tail_n)
    csv_path, md_path = write_summary(rows, output_dir, args.tail_n)
    fig_path = save_plot(rows, output_dir)

    print(f"Saved fixed-cache comparison results to: {output_dir}")
    print(f"Summary CSV: {csv_path}")
    print(f"Summary Markdown: {md_path}")
    print(f"Figure: {fig_path}")
    print("\nTail means:")
    for row in rows:
        print(
            f"  {row['label']:<12} "
            f"completed={float(row['episode_completed_mean']):.2f}, "
            f"hits={float(row['episode_hits_mean']):.2f}, "
            f"violations={float(row['episode_violation_steps_mean']):.2f}, "
            f"csafe={float(row['episode_avg_csafe_mean']):.2f}"
        )


if __name__ == "__main__":
    main()
