"""Plot hard-scope ablation in 3D metric space.

Input:
  ablation study/hard_scope_test_selected20/_summary_hard_scope_means.csv

Output:
  ablation study/hard_scope_test_selected20/hard_scope_3d.{png,pdf}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize


VARIANT_ORDER = ["hard_train_only", "hard_none", "hard_infer_only", "hard_both"]
VARIANT_LABEL = {
    "hard_train_only": "train-only",
    "hard_none": "none",
    "hard_infer_only": "infer-only",
    "hard_both": "train+infer",
}

METRICS = [
    ("baseline_deviation", "BD", "low"),
    ("cumulative_return", "Return", "high"),
    ("direction_reversal_rate", "DRR", "low"),
    ("unsupported_entry_rate", "UER", "low"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3D hard-scope ablation chart.")
    p.add_argument(
        "--input-csv",
        default="ablation study/hard_scope_test_selected20/_summary_hard_scope_means.csv",
        help="Input csv path.",
    )
    p.add_argument(
        "--output-prefix",
        default="ablation study/hard_scope_test_selected20/hard_scope_3d",
        help="Output prefix without extension.",
    )
    p.add_argument("--dpi", type=int, default=320)
    p.add_argument("--elev", type=float, default=24.0)
    p.add_argument("--azim", type=float, default=-58.0)
    p.add_argument("--bar-width", type=float, default=0.36, help="3D bar width on x axis.")
    p.add_argument("--bar-depth", type=float, default=0.36, help="3D bar depth on y axis.")
    p.add_argument(
        "--compress-alpha",
        type=float,
        default=1.8,
        help="Only for raw mode: visual height compression z=v/(1+alpha*v). Set 0 to disable.",
    )
    p.add_argument("--cmap", type=str, default="YlGnBu", help="Matplotlib colormap name for bar colors.")
    p.add_argument(
        "--pastel-strength",
        type=float,
        default=0.45,
        help="Blend ratio toward white for softer bars. 0=no blend, 1=all white.",
    )
    p.add_argument(
        "--scale",
        choices=["norm", "raw"],
        default="norm",
        help="norm: direction-aware min-max to [0,1]; raw: use original values.",
    )
    return p.parse_args()


def _normalize_directional(values: np.ndarray, direction: str) -> np.ndarray:
    arr = values.astype(float)
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if np.isclose(lo, hi):
        out = np.ones_like(arr)
    else:
        out = (arr - lo) / (hi - lo)
    if direction == "low":
        out = 1.0 - out
    return out


def _blend_to_white(colors: np.ndarray, strength: float) -> np.ndarray:
    s = np.clip(float(strength), 0.0, 1.0)
    base = colors[:, :3]
    out = (1.0 - s) * base + s * np.ones_like(base)
    if colors.shape[1] == 4:
        return np.column_stack([out, colors[:, 3]])
    return out


def _fmt_value(v: float, scale: str) -> str:
    if abs(v) < 5e-4:
        return "0"
    return f"{v:.2f}" if scale == "norm" else f"{v:.3f}"


def main() -> None:
    args = parse_args()
    in_path = Path(args.input_csv)
    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    df = df.set_index("variant").loc[VARIANT_ORDER].reset_index()

    x = np.arange(len(VARIANT_ORDER))
    y = np.arange(len(METRICS))
    xx, yy = np.meshgrid(x, y, indexing="ij")

    vals = np.zeros_like(xx, dtype=float)
    for i, v in enumerate(VARIANT_ORDER):
        row = df[df["variant"] == v].iloc[0]
        for j, (k, _, _) in enumerate(METRICS):
            vals[i, j] = float(row[k])

    # optional direction-aware normalization (higher is always better)
    if args.scale == "norm":
        vals_norm = np.zeros_like(vals, dtype=float)
        for j, (_, _, direction) in enumerate(METRICS):
            vals_norm[:, j] = _normalize_directional(vals[:, j], direction)
        vals_plot = vals_norm
    else:
        vals_plot = vals

    dx = float(args.bar_width)
    dy = float(args.bar_depth)
    xpos = xx.ravel() - dx / 2
    ypos = yy.ravel() - dy / 2
    zpos = np.zeros_like(xpos)
    if args.scale == "raw" and args.compress_alpha > 0:
        vals_draw = vals_plot / (1.0 + float(args.compress_alpha) * vals_plot)
    else:
        vals_draw = vals_plot
    dz = vals_draw.ravel()

    # color by value (higher is warmer)
    cmap = plt.get_cmap(args.cmap)
    norm = Normalize(vmin=float(vals_plot.min()), vmax=float(vals_plot.max()))
    colors = cmap(norm(vals_plot.ravel()))
    colors = _blend_to_white(colors, args.pastel_strength)

    fig = plt.figure(figsize=(11.2, 8.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.bar3d(
        xpos,
        ypos,
        zpos,
        dx,
        dy,
        dz,
        color=colors,
        edgecolor="#334155",
        linewidth=0.6,
        shade=False,
        alpha=0.90,
    )

    # labels
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABEL[v] for v in VARIANT_ORDER], fontsize=13, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([m for _, m, _ in METRICS], fontsize=13, fontweight="bold")
    ax.set_zlabel("", fontsize=16, labelpad=12)
    ax.set_xlabel("", fontsize=15, labelpad=12)
    ax.set_ylabel("", fontsize=15, labelpad=12)
    ax.tick_params(axis="z", labelsize=12)

    # numeric annotation
    zmax = max(1.0, float(vals_draw.max()))
    zcap = float(vals_draw.max()) * 1.14
    for xi in range(len(x)):
        for yi in range(len(y)):
            v_raw = vals_plot[xi, yi]
            v_draw = vals_draw[xi, yi]
            is_small = bool(args.scale == "raw" and abs(v_raw) < 0.012)
            if is_small:
                # near-zero labels are common; push them toward cell corners to avoid
                # collision with neighboring non-zero labels under 3D projection.
                if xi == 2 and yi == 0:
                    # Manual fix: BD@infer-only should stay on its own tiny bar.
                    xoff = 0.02
                    yoff = -0.02
                elif yi == 2:
                    # DRR near-zero labels: bias left to avoid covering nearby Return labels.
                    xoff = -0.20
                    yoff = 0.24
                elif yi >= 3:
                    # DRR/UER rows tend to project over Return labels at current view angle.
                    xoff = 0.28
                    yoff = 0.30
                else:
                    xoff = 0.22 if (yi % 2 == 0) else -0.22
                    yoff = 0.22 if (xi % 2 == 0) else -0.22
                tx = xi + xoff
                ty = yi + yoff
                tx = min(max(tx, -0.10), len(x) - 1 + 0.10)
                ty = min(max(ty, -0.10), len(y) - 1 + 0.10)
                zt = max(v_draw, 0.0) + 0.010 * zmax + 0.002 * ((xi + yi) % 2)
                fsize = 12.8
            else:
                zt = v_draw + 0.014 * zmax
                tx, ty = xi, yi
                fsize = 12.8
            zt = min(zt, zcap)
            ax.text(
                tx,
                ty,
                zt,
                _fmt_value(v_raw, args.scale),
                ha="center",
                va="bottom",
                fontsize=fsize,
                fontweight="semibold",
                color="#111827",
            )

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.66, pad=0.06)
    cbar.set_label("", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    ax.view_init(elev=args.elev, azim=args.azim)
    if args.scale == "norm":
        ax.set_zlim(0.0, 1.05)
    elif args.compress_alpha > 0:
        ax.set_zlabel("", fontsize=16, labelpad=12)
        ax.set_zlim(0.0, max(0.10, zcap + 0.012))
    ax.grid(True, linestyle="--", alpha=0.25)
    fig.subplots_adjust(left=0.02, right=0.94, top=0.98, bottom=0.03)

    png = out_prefix.with_suffix(".png")
    pdf = out_prefix.with_suffix(".pdf")
    fig.savefig(png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


if __name__ == "__main__":
    main()
