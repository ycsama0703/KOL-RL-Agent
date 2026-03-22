#!/usr/bin/env python3
"""Analyze betrayal forms by method (selected subset).

This script summarizes *how* each method tends to betray KOL intent.

Input:
- detailed compare CSV (one row per source/kol/method), e.g.
  benchmarks/compare/meta/selected20_all_methods_vs_baseline_detailed.csv

Outputs:
- betrayal_forms_by_method_source_raw.csv
- betrayal_forms_by_method_source_scaled.csv
- betrayal_forms_heatmap_scaled.png/.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_METHOD_ORDER = ["KICL", "AWAC", "IQL", "BC", "CQL", "TD3BC"]
FORM_ORDER = ["UER", "DRR", "BD", "CG"]
FORM_LABEL = {
    "UER": "Unsupported Entry (UER)",
    "DRR": "Direction Reversal (DRR)",
    "BD": "Behavior Deviation (BD)",
    "CG": "Correlation Gap (CG)",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-csv",
        default="benchmarks/compare/meta/selected20_all_methods_vs_baseline_detailed.csv",
        help="Detailed compare CSV.",
    )
    p.add_argument(
        "--output-dir",
        default="benchmarks/compare/analysis_betrayal_forms",
        help="Output directory.",
    )
    p.add_argument(
        "--methods",
        nargs="*",
        default=DEFAULT_METHOD_ORDER,
        help="Methods to include (order preserved).",
    )
    p.add_argument("--dpi", type=int, default=320)
    p.add_argument(
        "--cmap",
        default="YlGnBu",
        help="Matplotlib colormap for heatmap (e.g., YlGnBu, cividis, viridis).",
    )
    p.add_argument(
        "--annot-threshold",
        type=float,
        default=0.62,
        help="Switch annotation text color when cell value exceeds this threshold.",
    )
    p.add_argument(
        "--highlight-method",
        default="KICL",
        help="Method name to highlight in heatmap (row box + bold y-label).",
    )
    p.add_argument(
        "--highlight-color",
        default="#111111",
        help="Color of the highlight box.",
    )
    p.add_argument(
        "--highlight-style",
        choices=["label_only", "label_plus_marker", "band"],
        default="label_only",
        help="How to highlight the selected method row.",
    )
    p.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="Global font scale multiplier for better readability in paper layout.",
    )
    p.add_argument(
        "--colorbar-side",
        choices=["left", "right"],
        default="left",
        help="Place the heatmap colorbar on the left or right side.",
    )
    return p.parse_args()


def _compute_forms(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["UER"] = d["entry_violation_rate"].astype(float).clip(lower=0.0)
    d["DRR"] = d["reversal_rate"].astype(float).clip(lower=0.0)
    d["BD"] = d["mean_abs_deviation"].astype(float).clip(lower=0.0)
    d["CG"] = (1.0 - d["baseline_policy_corr"].astype(float)).clip(lower=0.0)
    return d


def _aggregate_raw(d: pd.DataFrame) -> pd.DataFrame:
    out = (
        d.groupby(["source", "method"], as_index=False)[FORM_ORDER]
        .mean()
        .sort_values(["source", "method"], kind="mergesort")
    )
    return out


def _scale_within_source(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for src, sdf in raw.groupby("source", sort=False):
        sdf = sdf.copy()
        for f in FORM_ORDER:
            mx = float(sdf[f].max())
            sdf[f"{f}_scaled"] = sdf[f] / mx if mx > 1e-12 else 0.0
        score_cols = [f"{f}_scaled" for f in FORM_ORDER]
        sdf["dominant_form"] = sdf[score_cols].idxmax(axis=1).str.replace("_scaled", "", regex=False)
        rows.append(sdf)
    return pd.concat(rows, ignore_index=True) if rows else raw.copy()


def _sorted_methods_present(raw: pd.DataFrame, wanted: list[str]) -> list[str]:
    present = set(raw["method"].tolist())
    ordered = [m for m in wanted if m in present]
    extras = sorted(m for m in present if m not in wanted)
    return ordered + extras


def _draw_heatmap(
    scaled: pd.DataFrame,
    methods: list[str],
    out_png: Path,
    out_pdf: Path,
    dpi: int,
    cmap: str,
    annot_threshold: float,
    highlight_method: str,
    highlight_color: str,
    highlight_style: str,
    font_scale: float,
    colorbar_side: str,
) -> None:
    sources = [s for s in ["x", "youtube"] if s in set(scaled["source"])]
    if not sources:
        raise RuntimeError("No supported source found in scaled table.")

    fig, axes = plt.subplots(1, len(sources), figsize=(8.2, 3.4), squeeze=False)
    axes = axes[0]

    for i, (ax, src) in enumerate(zip(axes, sources)):
        sdf = scaled[scaled["source"] == src].copy()
        sdf = sdf.set_index("method")
        mat = []
        idx = []
        for m in methods:
            if m in sdf.index:
                mat.append([float(sdf.loc[m, f"{f}_scaled"]) for f in FORM_ORDER])
                idx.append(m)
        if not mat:
            ax.set_axis_off()
            continue

        arr = np.asarray(mat, dtype=float)
        im = ax.imshow(arr, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title("X" if src == "x" else "YouTube", fontsize=10.5 * font_scale)
        ax.set_xticks(np.arange(len(FORM_ORDER)))
        ax.set_xticklabels(FORM_ORDER, fontsize=8.0 * font_scale)
        ax.set_yticks(np.arange(len(idx)))
        ax.set_yticklabels(idx, fontsize=8.0 * font_scale)
        if i > 0:
            ax.tick_params(axis="y", labelleft=False)

        # subtle cell borders for readability
        ax.set_xticks(np.arange(-0.5, len(FORM_ORDER), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(idx), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5, alpha=0.55)
        ax.tick_params(which="minor", bottom=False, left=False)

        highlight_row = idx.index(highlight_method) if highlight_method in idx else -1

        # numeric annotation
        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                val = arr[r, c]
                is_highlight_row = (r == highlight_row)
                ax.text(
                    c,
                    r,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=(8.2 if is_highlight_row else 6.7) * font_scale,
                    fontweight="bold" if is_highlight_row else "normal",
                    color="#1f2937" if val < annot_threshold else "white",
                )

        # highlight selected method row (subtle tint + label emphasis)
        if highlight_method in idx:
            hr = idx.index(highlight_method)
            if highlight_style == "band":
                ax.axhspan(
                    hr - 0.5,
                    hr + 0.5,
                    color=highlight_color,
                    alpha=0.10,
                    zorder=5,
                )
            if highlight_style == "label_plus_marker":
                ax.scatter(
                    [-0.62],
                    [hr],
                    s=28,
                    color=highlight_color,
                    zorder=7,
                    clip_on=False,
                )
            # bold y-label on first panel only (where labels are shown)
            if i == 0:
                for tick in ax.get_yticklabels():
                    if tick.get_text() == highlight_method:
                        tick.set_fontweight("bold")
                        tick.set_color(highlight_color)

    if colorbar_side == "left":
        # Move colorbar slightly inward as font grows, to avoid label clipping.
        extra = max(font_scale - 1.0, 0.0)
        cb_left = 0.08 + 0.03 * extra
        cb_w = 0.018
        plot_left = cb_left + cb_w + 0.085
        fig.subplots_adjust(left=plot_left, right=0.965, wspace=0.08, bottom=0.16)
        cax = fig.add_axes([cb_left, 0.18, cb_w, 0.67])  # [left, bottom, width, height]
    else:
        fig.subplots_adjust(right=0.885, wspace=0.08, bottom=0.16)
        cax = fig.add_axes([0.895, 0.18, 0.016, 0.67])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cax)
    cbar_label = "Scaled betrayal form intensity"
    cbar.set_label(cbar_label, fontsize=8.4 * font_scale)
    cbar.ax.tick_params(labelsize=7.5 * font_scale)
    if colorbar_side == "left":
        cbar.ax.yaxis.set_ticks_position("left")
        cbar.ax.yaxis.set_label_position("left")
    fig.supxlabel("Betrayal form", fontsize=9.0 * font_scale, y=0.01)
    fig.savefig(out_png, dpi=dpi)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    in_csv = Path(args.input_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    need = {
        "source",
        "method",
        "kol",
        "entry_violation_rate",
        "reversal_rate",
        "sign_agreement_rate",
        "mean_abs_deviation",
    }
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns in input CSV: {sorted(miss)}")

    d = df.copy()
    d = d[d["method"].isin(args.methods)].copy()
    d = d[d["method"].str.upper() != "BASELINE"].copy()
    d = _compute_forms(d)

    raw = _aggregate_raw(d)
    scaled = _scale_within_source(raw)
    methods = _sorted_methods_present(raw, args.methods)

    raw_out = out_dir / "betrayal_forms_by_method_source_raw.csv"
    scaled_out = out_dir / "betrayal_forms_by_method_source_scaled.csv"
    raw.to_csv(raw_out, index=False)
    scaled.to_csv(scaled_out, index=False)

    png = out_dir / "betrayal_forms_heatmap_scaled.png"
    pdf = out_dir / "betrayal_forms_heatmap_scaled.pdf"
    _draw_heatmap(
        scaled,
        methods,
        png,
        pdf,
        dpi=args.dpi,
        cmap=args.cmap,
        annot_threshold=args.annot_threshold,
        highlight_method=args.highlight_method,
        highlight_color=args.highlight_color,
        highlight_style=args.highlight_style,
        font_scale=args.font_scale,
        colorbar_side=args.colorbar_side,
    )

    # small markdown summary
    md = out_dir / "betrayal_forms_summary.md"
    lines = []
    lines.append("# Betrayal Form Profile")
    lines.append("")
    lines.append("Forms:")
    lines.append("- `UER`: unsupported entry rate")
    lines.append("- `DRR`: direction reversal rate")
    lines.append("- `BD`: mean absolute behavior deviation")
    lines.append("- `CG`: correlation gap (= 1 - baseline policy correlation)")
    lines.append("")
    for src in [s for s in ["x", "youtube"] if s in set(scaled["source"])]:
        lines.append(f"## {src}")
        sdf = scaled[scaled["source"] == src].copy().sort_values("method")
        for _, r in sdf.iterrows():
            lines.append(
                f"- `{r['method']}` dominant form: **{r['dominant_form']}** "
                f"(UER={r['UER_scaled']:.2f}, DRR={r['DRR_scaled']:.2f}, "
                f"BD={r['BD_scaled']:.2f}, CG={r['CG_scaled']:.2f})"
            )
        lines.append("")
    md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {raw_out}")
    print(f"Saved: {scaled_out}")
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")
    print(f"Saved: {md}")


if __name__ == "__main__":
    main()
