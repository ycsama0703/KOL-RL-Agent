"""Plot five-point ablation deltas as a 3D bar chart.

X axis: methods
Y axis: metrics (Return / Sharpe / BD)
Z axis: delta value vs baseline
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb


DEFAULT_ORDER = ["WO_RL_COMPLETION", "WO_REGIME_SPLIT", "KICL", "WO_HARD"]
LABELS = {
    "WO_RL_COMPLETION": "WO-RL-C",
    "WO_REGIME_SPLIT": "WO-RS",
    "KICL": "KICL",
    "WO_HARD": "WO-H",
}
METRIC_COLS = ["event_return_mean", "event_sharpe_mean", "BD_mean"]
METRIC_LABELS = ["ΔReturn", "ΔSharpe", "ΔBD"]
METHOD_COLORS = {
    "BASELINE": "#d8e6f5",
    "WO_RL_COMPLETION": "#bfd3ea",
    "WO_REGIME_SPLIT": "#9fc0e0",
    "KICL": "#f5c89a",
    "WO_HARD": "#e7a3a3",
}
METRIC_SHADE = [1.00, 0.92, 0.84]  # slightly darker by metric for separation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot five-point 3D bars.")
    p.add_argument(
        "--input-csv",
        default="ablation study/five_point_compare/five_point_summary_overall.csv",
        help="Input summary csv containing BASELINE and ablation rows.",
    )
    p.add_argument(
        "--output-prefix",
        default="ablation study/five_point_compare/five_point_delta3_3d",
        help="Output prefix (without extension).",
    )
    p.add_argument(
        "--include-baseline",
        action="store_true",
        help="Include baseline (all-zero deltas) in the method axis.",
    )
    p.add_argument(
        "--bar-width",
        type=float,
        default=0.34,
        help="3D bar width/depth. Smaller values produce thinner bars.",
    )
    p.add_argument(
        "--focus-progressive",
        action="store_true",
        help="Only plot WO-RL-C, WO-RS, and KICL (exclude WO-H).",
    )
    p.add_argument("--dpi", type=int, default=260)
    return p.parse_args()


def _shade(color_hex: str, factor: float) -> tuple[float, float, float]:
    r, g, b = to_rgb(color_hex)
    return (min(1.0, r * factor), min(1.0, g * factor), min(1.0, b * factor))


def main() -> None:
    args = parse_args()
    in_csv = Path(args.input_csv)
    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv).set_index("method")
    missing_required = [m for m in ["BASELINE"] + DEFAULT_ORDER if m not in df.index]
    if missing_required:
        raise RuntimeError(f"Missing rows in input csv: {missing_required}")

    ordered = list(DEFAULT_ORDER)
    if args.focus_progressive:
        ordered = ["WO_RL_COMPLETION", "WO_REGIME_SPLIT", "KICL"]

    methods = ["BASELINE"] + ordered if args.include_baseline else list(ordered)
    method_labels = ["Baseline"] + [LABELS[m] for m in DEFAULT_ORDER] if args.include_baseline else [LABELS[m] for m in DEFAULT_ORDER]
    if args.focus_progressive:
        method_labels = ["Baseline"] + [LABELS[m] for m in ordered] if args.include_baseline else [LABELS[m] for m in ordered]

    base_vals = df.loc["BASELINE", METRIC_COLS]
    delta = pd.DataFrame(index=methods, columns=METRIC_LABELS, dtype=float)
    for col, label in zip(METRIC_COLS, METRIC_LABELS):
        if col not in df.columns:
            raise RuntimeError(f"Missing column in csv: {col}")
        if "BASELINE" in methods:
            delta.loc["BASELINE", label] = 0.0
        for m in ordered:
            if m in methods:
                delta.loc[m, label] = float(df.loc[m, col] - base_vals[col])

    # build bar coordinates
    x_idx = np.arange(len(methods))
    y_idx = np.arange(len(METRIC_LABELS))
    xx, yy = np.meshgrid(x_idx, y_idx, indexing="ij")
    x = xx.ravel().astype(float)
    y = yy.ravel().astype(float)
    z = np.zeros_like(x)
    dz = np.array([delta.iloc[i, j] for i in range(len(methods)) for j in range(len(METRIC_LABELS))], dtype=float)

    bw = float(args.bar_width)
    dx = np.full_like(x, bw, dtype=float)
    dy = np.full_like(y, bw, dtype=float)

    colors = []
    for i in range(len(methods)):
        m = methods[i]
        base_c = METHOD_COLORS.get(m, "#b0c4de")
        for j in range(len(METRIC_LABELS)):
            colors.append(_shade(base_c, METRIC_SHADE[j]))

    fig = plt.figure(figsize=(10.8, 6.4))
    ax = fig.add_subplot(111, projection="3d")

    # zero plane for easier sign reading
    x_plane = np.linspace(-0.2, len(methods) - 0.2, 2)
    y_plane = np.linspace(-0.2, len(METRIC_LABELS) - 0.2, 2)
    xg, yg = np.meshgrid(x_plane, y_plane)
    zg = np.zeros_like(xg)
    ax.plot_surface(xg, yg, zg, alpha=0.12, color="#9ca3af", linewidth=0, shade=False)

    ax.bar3d(
        x - dx / 2,
        y - dy / 2,
        z,
        dx,
        dy,
        dz,
        color=colors,
        edgecolor="#5b6b7f",
        linewidth=0.6,
        shade=True,
        alpha=0.88,
    )

    # Light trend guides from front->back (method progression) for each metric
    x_idx = np.arange(len(methods))
    for j, metric in enumerate(METRIC_LABELS):
        zs = [float(delta.iloc[i, j]) for i in range(len(methods))]
        ax.plot(
            x_idx,
            np.full_like(x_idx, j, dtype=float),
            zs,
            color="#64748b",
            linestyle="--",
            linewidth=1.0,
            alpha=0.65,
        )

    # value labels
    for xv, yv, zv in zip(x, y, dz):
        z_text = zv + (0.012 if zv >= 0 else -0.02)
        va = "bottom" if zv >= 0 else "top"
        ax.text(xv, yv, z_text, f"{zv:+.3f}", ha="center", va=va, fontsize=8.5, color="#1f2937")

    ax.set_xticks(x_idx)
    ax.set_xticklabels(method_labels, fontsize=10)
    ax.set_yticks(y_idx)
    ax.set_yticklabels(METRIC_LABELS, fontsize=10)
    ax.set_zlabel("Delta vs Baseline", fontsize=11, labelpad=10)
    ax.set_title("Progressive ablation path in 3D metric space", fontsize=16, pad=12, weight="semibold")

    ax.view_init(elev=22, azim=-42)
    ax.set_box_aspect((1.8, 1.0, 0.9))
    ax.grid(True, alpha=0.35)

    # No colorbar: color encodes method progression, not value sign.

    png = out_prefix.with_suffix(".png")
    pdf = out_prefix.with_suffix(".pdf")
    plt.tight_layout()
    plt.savefig(png, dpi=args.dpi, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png}")
    print(f"Saved: {pdf}")
    print(delta.round(6).to_string())


if __name__ == "__main__":
    main()
