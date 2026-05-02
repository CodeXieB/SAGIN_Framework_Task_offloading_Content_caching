"""
Plot safety-performance tradeoff for SAGIN PPO experiments.

This script is intentionally standalone: it only reads metrics.npz files from
artifact directories and does not modify any existing training or plotting code.

Example:
    python generate_safety_performance_tradeoff_plot.py

    python generate_safety_performance_tradeoff_plot.py --schemes baseline safe_a safe_b safe_c
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


SCHEME_META: Dict[str, Tuple[str, str, str]] = {
    "baseline": ("Baseline PPO", "#1f77b4", "o"),
    "safe_a": ("Safe-A", "#d62728", "s"),
    "safe_b": ("Safe-B", "#2ca02c", "^"),
    "safe_c": ("Safe-C", "#ff7f0e", "D"),
    "curriculum_safe": ("Curriculum-Safe", "#9467bd", "P"),
    "curriculum_safe_v2": ("Curriculum-Safe-v2", "#8c564b", "X"),
    "curriculum_safe_v3": ("Curriculum-Safe-v3", "#17becf", "*"),
}

LABEL_OFFSETS: Dict[str, Tuple[int, int, str]] = {
    "baseline": (10, 8, "left"),
    "safe_a": (-58, -22, "right"),
    "safe_b": (28, 30, "left"),
    "safe_c": (10, -18, "left"),
    "curriculum_safe": (10, 10, "left"),
    "curriculum_safe_v2": (10, 10, "left"),
    "curriculum_safe_v3": (10, 10, "left"),
}

POINT_TAGS: Dict[str, str] = {
    "baseline": "B",
    "safe_a": "A",
    "safe_b": "B",
    "safe_c": "C",
    "curriculum_safe": "CL",
    "curriculum_safe_v2": "C2",
    "curriculum_safe_v3": "C3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create safety-performance tradeoff plots from experiment metrics."
    )
    parser.add_argument("--artifact-root", default="artifacts", help="Root artifact directory.")
    parser.add_argument(
        "--schemes",
        nargs="+",
        default=["baseline", "safe_a", "safe_b", "safe_c"],
        help="Experiment subdirectories under artifact-root.",
    )
    parser.add_argument("--tail-n", type=int, default=50, help="Use the last N episodes.")
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output figure path. If omitted, the figure is saved under "
            "artifacts/generated_results/<timestamp>/."
        ),
    )
    parser.add_argument(
        "--performance-metric",
        default="episode_completed",
        choices=["episode_completed", "episode_hits", "episode_avg_rperf", "episode_rewards"],
        help="Y-axis performance metric.",
    )
    parser.add_argument(
        "--safety-metric",
        default="episode_violation_steps",
        choices=["episode_violation_steps", "episode_avg_csafe", "episode_avg_failed_offloads"],
        help="X-axis safety cost metric. Lower is better.",
    )
    return parser.parse_args()


def tail_mean(values: np.ndarray, tail_n: int) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    tail = values[-tail_n:] if values.size >= tail_n else values
    return float(np.nanmean(tail))


def load_point(
    artifact_root: Path,
    scheme: str,
    tail_n: int,
    safety_metric: str,
    performance_metric: str,
) -> Tuple[float, float]:
    metrics_path = artifact_root / scheme / "metrics.npz"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    metrics = np.load(metrics_path, allow_pickle=False)
    missing = [k for k in [safety_metric, performance_metric] if k not in metrics]
    if missing:
        raise KeyError(f"{metrics_path} is missing metrics: {', '.join(missing)}")

    safety = tail_mean(metrics[safety_metric], tail_n)
    performance = tail_mean(metrics[performance_metric], tail_n)
    return safety, performance


def pretty_metric_name(metric: str) -> str:
    names = {
        "episode_completed": "Completed Tasks",
        "episode_hits": "Cache Hits",
        "episode_avg_rperf": "Average Performance Reward",
        "episode_rewards": "Episode Reward",
        "episode_violation_steps": "Violation Steps",
        "episode_avg_csafe": "Average Safety Cost",
        "episode_avg_failed_offloads": "Average Failed Offloads",
    }
    return names.get(metric, metric)


def _svg_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _save_svg_fallback(
    output_path: Path,
    points: List[Tuple[str, str, str, str, float, float]],
    tail_n: int,
    safety_metric: str,
    performance_metric: str,
) -> Path:
    svg_path = output_path.with_suffix(".svg")
    width, height = 920, 580
    left, right, top, bottom = 95, 35, 60, 85
    plot_w = width - left - right
    plot_h = height - top - bottom

    xs = np.asarray([p[4] for p in points], dtype=np.float64)
    ys = np.asarray([p[5] for p in points], dtype=np.float64)
    x_min, x_max = float(np.nanmin(xs)), float(np.nanmax(xs))
    y_min, y_max = float(np.nanmin(ys)), float(np.nanmax(ys))
    x_pad = max((x_max - x_min) * 0.12, 1e-6)
    y_pad = max((y_max - y_min) * 0.12, 1e-6)
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    def sx(x: float) -> float:
        return left + (x - x_min) / max(x_max - x_min, 1e-9) * plot_w

    def sy(y: float) -> float:
        return top + plot_h - (y - y_min) / max(y_max - y_min, 1e-9) * plot_h

    def tick_values(v_min: float, v_max: float, n: int = 5) -> np.ndarray:
        return np.linspace(v_min, v_max, n)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222} .grid{stroke:#ddd;stroke-width:1} .axis{stroke:#222;stroke-width:1.3}</style>',
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" font-size="18">Safety-Performance Tradeoff (tail {tail_n} episodes)</text>',
    ]

    for x in tick_values(x_min, x_max):
        px = sx(float(x))
        lines.append(f'<line class="grid" x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{top + plot_h}"/>')
        lines.append(f'<text x="{px:.1f}" y="{top + plot_h + 24}" text-anchor="middle" font-size="11">{x:.2f}</text>')
    for y in tick_values(y_min, y_max):
        py = sy(float(y))
        lines.append(f'<line class="grid" x1="{left}" y1="{py:.1f}" x2="{left + plot_w}" y2="{py:.1f}"/>')
        lines.append(f'<text x="{left - 10}" y="{py + 4:.1f}" text-anchor="end" font-size="11">{y:.1f}</text>')

    lines.extend([
        f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>',
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 26}" text-anchor="middle" font-size="13">{_svg_escape(pretty_metric_name(safety_metric))} (lower is better)</text>',
        f'<text x="22" y="{top + plot_h / 2:.1f}" transform="rotate(-90 22 {top + plot_h / 2:.1f})" text-anchor="middle" font-size="13">{_svg_escape(pretty_metric_name(performance_metric))} (higher is better)</text>',
        f'<text x="{left + 12}" y="{top + 18}" font-size="12" fill="#2ca02c">better</text>',
        f'<line x1="{left + 115}" y1="{top + 105}" x2="{left + 28}" y2="{top + 30}" stroke="#2ca02c" stroke-width="2" marker-end="url(#arrow)"/>',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#2ca02c"/></marker></defs>',
    ])

    marker_shape = {
        "o": "circle",
        "s": "square",
        "^": "triangle",
        "D": "diamond",
        "P": "plus",
        "X": "cross",
        "*": "star",
    }
    for scheme, label, color, marker, safety, performance in points:
        px, py = sx(safety), sy(performance)
        shape = marker_shape.get(marker, "circle")
        if shape == "square":
            lines.append(f'<rect x="{px - 7:.1f}" y="{py - 7:.1f}" width="14" height="14" fill="{color}" stroke="#111"/>')
        elif shape == "triangle":
            lines.append(f'<path d="M{px:.1f},{py - 9:.1f} L{px - 9:.1f},{py + 8:.1f} L{px + 9:.1f},{py + 8:.1f} Z" fill="{color}" stroke="#111"/>')
        elif shape == "diamond":
            lines.append(f'<path d="M{px:.1f},{py - 10:.1f} L{px + 10:.1f},{py:.1f} L{px:.1f},{py + 10:.1f} L{px - 10:.1f},{py:.1f} Z" fill="{color}" stroke="#111"/>')
        else:
            lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="8" fill="{color}" stroke="#111"/>')
        tag = POINT_TAGS.get(scheme, "")
        if tag:
            lines.append(
                f'<text x="{px:.1f}" y="{py + 4:.1f}" text-anchor="middle" '
                f'font-size="10" font-weight="700" fill="white">{_svg_escape(tag)}</text>'
            )

        dx, dy, anchor = LABEL_OFFSETS.get(scheme, (11, -8, "left"))
        label_x = px + dx
        label_y = py + dy
        if scheme in {"safe_a", "safe_b"}:
            lines.append(
                f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{label_x:.1f}" y2="{label_y:.1f}" '
                f'stroke="{color}" stroke-width="1" stroke-dasharray="3,3" opacity="0.8"/>'
            )
        text_anchor = "end" if anchor == "right" else "start"
        lines.append(
            f'<text x="{label_x:.1f}" y="{label_y:.1f}" text-anchor="{text_anchor}" '
            f'font-size="12" font-weight="600">{_svg_escape(label)}</text>'
        )

    lines.append("</svg>")
    svg_path.write_text("\n".join(lines), encoding="utf-8")
    return svg_path


def plot_tradeoff(args: argparse.Namespace) -> None:
    artifact_root = Path(args.artifact_root)
    if args.output:
        output_path = Path(args.output)
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = (
            artifact_root
            / "generated_results"
            / run_id
            / "safety_performance_tradeoff.png"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    points = []
    for scheme in args.schemes:
        safety, performance = load_point(
            artifact_root,
            scheme,
            args.tail_n,
            args.safety_metric,
            args.performance_metric,
        )
        label, color, marker = SCHEME_META.get(scheme, (scheme, "#7f7f7f", "o"))
        points.append((scheme, label, color, marker, safety, performance))

    if not HAS_MPL:
        svg_path = _save_svg_fallback(
            output_path,
            points,
            args.tail_n,
            args.safety_metric,
            args.performance_metric,
        )
        print(f"matplotlib is not installed; saved SVG fallback: {svg_path}")
        print_points(points, args.safety_metric, args.performance_metric)
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for scheme, label, color, marker, safety, performance in points:
        ax.scatter(
            safety,
            performance,
            s=140,
            color=color,
            marker=marker,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
            label=label,
            zorder=3,
        )
        tag = POINT_TAGS.get(scheme, "")
        if tag:
            ax.text(
                safety,
                performance,
                tag,
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color="white",
                zorder=4,
            )
        dx, dy, anchor = LABEL_OFFSETS.get(scheme, (8, 6, "left"))
        ax.annotate(
            label,
            (safety, performance),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9,
            fontweight="semibold",
            ha="right" if anchor == "right" else "left",
            arrowprops=(
                dict(arrowstyle="-", color=color, lw=0.9, linestyle="--", alpha=0.8)
                if scheme in {"safe_a", "safe_b"}
                else None
            ),
        )

    ax.set_xlabel(f"{pretty_metric_name(args.safety_metric)} (lower is better)")
    ax.set_ylabel(f"{pretty_metric_name(args.performance_metric)} (higher is better)")
    ax.set_title(f"Safety-Performance Tradeoff (tail {args.tail_n} episodes)")
    ax.grid(True, alpha=0.28, zorder=0)
    ax.legend(loc="best", frameon=True)

    # Highlight the desirable direction in the figure without relying on Chinese fonts.
    ax.annotate(
        "better",
        xy=(0.06, 0.92),
        xycoords="axes fraction",
        fontsize=10,
        color="#2ca02c",
        ha="left",
        va="center",
    )
    ax.annotate(
        "",
        xy=(0.05, 0.88),
        xytext=(0.18, 0.72),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.5),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved tradeoff plot: {output_path}")
    print_points(points, args.safety_metric, args.performance_metric)


def print_points(
    points: List[Tuple[str, str, str, str, float, float]],
    safety_metric: str,
    performance_metric: str,
) -> None:
    print("Tail means:")
    for scheme, label, _, _, safety, performance in points:
        print(
            f"  {label:<20} "
            f"{safety_metric}={safety:.4f}, "
            f"{performance_metric}={performance:.4f}"
        )


def main() -> None:
    args = parse_args()
    plot_tradeoff(args)


if __name__ == "__main__":
    main()
