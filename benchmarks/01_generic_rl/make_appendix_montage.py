#!/usr/bin/env python3
"""Build appendix montage images from per-KOL comparison plots.

Default behavior:
- Read selected KOLs from:
  - benchmarks/compare/meta/kicl_top10_vs_baseline_youtube.csv
  - benchmarks/compare/meta/kicl_top10_vs_baseline_x.csv
- Read plot images from:
  - benchmarks/compare/canonical_all/<source>/<kol>/<image_name>
- Output one combined 20-panel figure, and optionally per-source figures.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


SourceKol = Tuple[str, str]
METHOD_ORDER = ["KICL", "BC", "IQL", "CQL", "TD3BC", "AWAC"]
METHOD_COLORS = {
    "KICL": "#F39C12",
    "BC": "#8C564B",
    "IQL": "#2CA02C",
    "CQL": "#D62728",
    "TD3BC": "#9467BD",
    "AWAC": "#17BECF",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=Path("benchmarks/compare/canonical_all"),
        help="Root directory containing canonical compare outputs.",
    )
    parser.add_argument(
        "--youtube-selection-csv",
        type=Path,
        default=Path("benchmarks/compare/meta/kicl_top10_vs_baseline_youtube.csv"),
        help="CSV with selected YouTube KOLs. Must contain a 'kol' column.",
    )
    parser.add_argument(
        "--x-selection-csv",
        type=Path,
        default=Path("benchmarks/compare/meta/kicl_top10_vs_baseline_x.csv"),
        help="CSV with selected X KOLs. Must contain a 'kol' column.",
    )
    parser.add_argument(
        "--sources",
        type=str,
        choices=["all", "youtube", "x"],
        default="all",
        help="Which source selection(s) to render.",
    )
    parser.add_argument(
        "--image-name",
        type=str,
        default="event_equity_compare.png",
        help="Image filename to pick inside each KOL folder.",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        choices=["image", "csv"],
        default="csv",
        help="Render by tiling images, or redraw subplots from CSV curves.",
    )
    parser.add_argument(
        "--curve-csv-name",
        type=str,
        default="event_equity_compare.csv",
        help="CSV filename to load when --render-mode csv.",
    )
    parser.add_argument(
        "--trim-flat-tail-days",
        type=int,
        default=0,
        help=(
            "If >0, stop plotting a method when its ending flat tail length "
            "is at least this many days (for better readability)."
        ),
    )
    parser.add_argument(
        "--trim-flat-tail-eps",
        type=float,
        default=1e-12,
        help="Tolerance when detecting flat tails.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/compare/appendix"),
        help="Output directory for montage images.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="selected20",
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=5,
        help="Number of columns for the combined montage.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=0,
        help="Optional fixed number of rows. 0 means auto from item count and cols.",
    )
    parser.add_argument(
        "--cell-width",
        type=float,
        default=4.6,
        help="Per-cell width in inches.",
    )
    parser.add_argument(
        "--cell-height",
        type=float,
        default=3.2,
        help="Per-cell height in inches.",
    )
    parser.add_argument(
        "--fig-width-cm",
        type=float,
        default=0.0,
        help="If >0, override figure width in cm.",
    )
    parser.add_argument(
        "--fig-height-cm",
        type=float,
        default=0.0,
        help="If >0, override figure height in cm.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Figure DPI.",
    )
    parser.add_argument(
        "--x-max-ticks",
        type=int,
        default=4,
        help="Maximum number of x-axis date ticks per subplot (csv render mode).",
    )
    parser.add_argument(
        "--label-fontsize",
        type=float,
        default=8.0,
        help="KOL label font size inside each subplot.",
    )
    parser.add_argument(
        "--tick-fontsize",
        type=float,
        default=6.6,
        help="Axis tick font size for each subplot.",
    )
    parser.add_argument(
        "--legend-fontsize",
        type=float,
        default=8.2,
        help="Global legend font size.",
    )
    parser.add_argument(
        "--legend-ncol",
        type=int,
        default=6,
        help="Number of legend columns.",
    )
    parser.add_argument(
        "--legend-y",
        type=float,
        default=0.01,
        help="Global legend y anchor in figure coordinates.",
    )
    parser.add_argument(
        "--tight-left",
        type=float,
        default=0.015,
        help="tight_layout left bound (0~1).",
    )
    parser.add_argument(
        "--tight-bottom",
        type=float,
        default=0.055,
        help="tight_layout bottom bound (0~1).",
    )
    parser.add_argument(
        "--tight-right",
        type=float,
        default=0.995,
        help="tight_layout right bound (0~1).",
    )
    parser.add_argument(
        "--tight-top",
        type=float,
        default=0.97,
        help="tight_layout top bound (0~1).",
    )
    parser.add_argument(
        "--split-by-source",
        action="store_true",
        help="Also emit one montage for each source (youtube/x).",
    )
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="Only emit source montages (youtube/x), skip combined montage.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Appendix: Per-KOL Equity Comparison",
        help="Combined montage title.",
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Disable combined title.",
    )
    return parser.parse_args()


def read_kols(csv_path: Path) -> List[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Selection CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "kol" not in df.columns:
        raise ValueError(f"'kol' column missing in {csv_path}")
    return [str(v) for v in df["kol"].dropna().tolist()]


def build_selection(youtube_csv: Path, x_csv: Path, sources: str) -> List[SourceKol]:
    youtube_kols = read_kols(youtube_csv)
    x_kols = read_kols(x_csv)
    selected: List[SourceKol] = []
    # Keep source blocks ordered for easier appendix reading.
    if sources in ("all", "youtube"):
        selected.extend([("youtube", k) for k in youtube_kols])
    if sources in ("all", "x"):
        selected.extend([("x", k) for k in x_kols])
    return selected


def locate_artifacts(
    canonical_root: Path, selected: Sequence[SourceKol], artifact_name: str
) -> Tuple[List[Tuple[str, str, Path]], List[SourceKol]]:
    found: List[Tuple[str, str, Path]] = []
    missing: List[SourceKol] = []
    for source, kol in selected:
        artifact_path = canonical_root / source / kol / artifact_name
        if artifact_path.exists():
            found.append((source, kol, artifact_path))
        else:
            missing.append((source, kol))
    return found, missing


def _grid_shape(count: int, cols: int, rows: int = 0) -> Tuple[int, int]:
    cols = max(1, cols)
    if rows > 0:
        if rows * cols < count:
            raise ValueError(
                f"Grid too small: rows*cols={rows*cols}, items={count}. "
                "Increase --rows or --cols."
            )
        return rows, cols
    rows = int(math.ceil(count / cols))
    return rows, cols


def _axes_to_list(axes, rows: int, cols: int):
    if rows == 1 and cols == 1:
        return [axes]
    if rows == 1:
        return list(axes)
    if cols == 1:
        return [axes[r] for r in range(rows)]
    return [axes[r, c] for r in range(rows) for c in range(cols)]


def make_montage(
    items: Sequence[Tuple[str, str, Path]],
    output_path: Path,
    cols: int,
    rows: int,
    cell_width: float,
    cell_height: float,
    dpi: int,
    title: str | None,
    fig_width_cm: float,
    fig_height_cm: float,
    tight_left: float,
    tight_bottom: float,
    tight_right: float,
    tight_top: float,
) -> None:
    if not items:
        raise ValueError("No items to plot.")

    rows, cols = _grid_shape(len(items), cols, rows)
    if fig_width_cm > 0 and fig_height_cm > 0:
        figsize = (fig_width_cm / 2.54, fig_height_cm / 2.54)
    else:
        figsize = (cols * cell_width, rows * cell_height)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes_list = _axes_to_list(axes, rows, cols)

    for ax in axes_list:
        ax.axis("off")

    for idx, (source, kol, img_path) in enumerate(items):
        ax = axes_list[idx]
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.set_title(f"{source}/{kol}", fontsize=9, pad=5)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=11.5, y=0.992)
    fig.tight_layout(rect=(tight_left, tight_bottom, tight_right, tight_top))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _plot_curve_panel(
    ax,
    source: str,
    kol: str,
    csv_path: Path,
    show_source_prefix: bool,
    trim_flat_tail_days: int,
    trim_flat_tail_eps: float,
    x_max_ticks: int,
    label_fontsize: float,
    tick_fontsize: float,
) -> List[Tuple[object, str]]:
    df = pd.read_csv(csv_path)
    if "date" not in df.columns:
        raise ValueError(f"'date' column not found in {csv_path}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    method_cols = [c for c in df.columns if c != "date"]
    ordered = [m for m in METHOD_ORDER if m in method_cols] + [
        m for m in method_cols if m not in METHOD_ORDER
    ]

    handles: List[Tuple[object, str]] = []
    for method in ordered:
        color = METHOD_COLORS.get(method, None)
        lw = 2.2 if method == "KICL" else 1.2
        y = df[method].astype(float).copy()
        if trim_flat_tail_days > 0 and len(y) >= trim_flat_tail_days:
            last = float(y.iloc[-1])
            n = 0
            # trailing points equal to last value
            for v in y.iloc[::-1]:
                if np.isnan(v) or abs(float(v) - last) > trim_flat_tail_eps:
                    break
                n += 1
            if n >= trim_flat_tail_days:
                y.iloc[len(y) - n :] = np.nan
        line = ax.plot(
            df["date"],
            y,
            label=method,
            color=color,
            linewidth=lw,
            alpha=0.95,
        )[0]
        handles.append((line, method))

    label = f"{source}/{kol}" if show_source_prefix else kol
    ax.text(
        0.02,
        0.96,
        label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=label_fontsize,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.25)
    # Keep date labels readable in dense 5x2 appendix grids.
    dmin = df["date"].min()
    dmax = df["date"].max()
    span_months = max(1, (dmax.year - dmin.year) * 12 + (dmax.month - dmin.month) + 1)
    max_ticks = max(2, x_max_ticks)
    month_interval = max(1, int(math.ceil(span_months / max_ticks)))
    locator = mdates.MonthLocator(interval=month_interval)
    formatter = mdates.DateFormatter("%Y-%m")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis="x", labelsize=tick_fontsize, rotation=0)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    return handles


def make_curve_grid(
    items: Sequence[Tuple[str, str, Path]],
    output_path: Path,
    cols: int,
    rows: int,
    cell_width: float,
    cell_height: float,
    dpi: int,
    title: str | None,
    show_source_prefix: bool,
    trim_flat_tail_days: int,
    trim_flat_tail_eps: float,
    x_max_ticks: int,
    fig_width_cm: float,
    fig_height_cm: float,
    label_fontsize: float,
    tick_fontsize: float,
    legend_fontsize: float,
    legend_ncol: int,
    legend_y: float,
    tight_left: float,
    tight_bottom: float,
    tight_right: float,
    tight_top: float,
) -> None:
    if not items:
        raise ValueError("No items to plot.")

    rows, cols = _grid_shape(len(items), cols, rows)
    if fig_width_cm > 0 and fig_height_cm > 0:
        figsize = (fig_width_cm / 2.54, fig_height_cm / 2.54)
    else:
        figsize = (cols * cell_width, rows * cell_height)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes_list = _axes_to_list(axes, rows, cols)

    for ax in axes_list:
        ax.set_visible(False)

    legend_map = {}
    for idx, (source, kol, csv_path) in enumerate(items):
        ax = axes_list[idx]
        ax.set_visible(True)
        pairs = _plot_curve_panel(
            ax=ax,
            source=source,
            kol=kol,
            csv_path=csv_path,
            show_source_prefix=show_source_prefix,
            trim_flat_tail_days=trim_flat_tail_days,
            trim_flat_tail_eps=trim_flat_tail_eps,
            x_max_ticks=x_max_ticks,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
        )
        for handle, label in pairs:
            if label not in legend_map:
                legend_map[label] = handle

    if title:
        fig.suptitle(title, fontsize=14, y=0.995)

    legend_labels = [m for m in METHOD_ORDER if m in legend_map] + [
        m for m in legend_map.keys() if m not in METHOD_ORDER
    ]
    legend_handles = [legend_map[m] for m in legend_labels]
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=min(max(1, legend_ncol), len(legend_labels)),
        frameon=True,
        fontsize=legend_fontsize,
        bbox_to_anchor=(0.5, legend_y),
    )
    fig.tight_layout(rect=(tight_left, tight_bottom, tight_right, tight_top))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def write_manifest(
    output_path: Path,
    items: Iterable[Tuple[str, str, Path]],
    missing: Iterable[SourceKol],
    image_name: str,
) -> None:
    rows = []
    for source, kol, img_path in items:
        rows.append(
            {
                "source": source,
                "kol": kol,
                "status": "found",
                "image_name": image_name,
                "image_path": str(img_path),
            }
        )
    for source, kol in missing:
        rows.append(
            {
                "source": source,
                "kol": kol,
                "status": "missing",
                "image_name": image_name,
                "image_path": "",
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    if args.source_only and not args.split_by_source:
        raise ValueError("--source-only requires --split-by-source.")

    selected = build_selection(args.youtube_selection_csv, args.x_selection_csv, args.sources)
    artifact_name = args.image_name if args.render_mode == "image" else args.curve_csv_name
    found, missing = locate_artifacts(args.canonical_root, selected, artifact_name)

    if not found:
        raise RuntimeError(
            f"No artifacts found. Check canonical root and artifact name ({artifact_name})."
        )

    stem = Path(artifact_name).stem
    combined_out = args.output_dir / f"{args.output_prefix}_{stem}_montage.png"
    if not args.source_only:
        combined_title = None if args.no_title else args.title
        if args.render_mode == "image":
            make_montage(
                items=found,
                output_path=combined_out,
                cols=args.cols,
                rows=args.rows,
                cell_width=args.cell_width,
                cell_height=args.cell_height,
                dpi=args.dpi,
                title=combined_title,
                fig_width_cm=args.fig_width_cm,
                fig_height_cm=args.fig_height_cm,
                tight_left=args.tight_left,
                tight_bottom=args.tight_bottom,
                tight_right=args.tight_right,
                tight_top=args.tight_top,
            )
        else:
            make_curve_grid(
                items=found,
                output_path=combined_out,
                cols=args.cols,
                rows=args.rows,
                cell_width=args.cell_width,
                cell_height=args.cell_height,
                dpi=args.dpi,
                title=combined_title,
                show_source_prefix=True,
                trim_flat_tail_days=args.trim_flat_tail_days,
                trim_flat_tail_eps=args.trim_flat_tail_eps,
                x_max_ticks=args.x_max_ticks,
                fig_width_cm=args.fig_width_cm,
                fig_height_cm=args.fig_height_cm,
                label_fontsize=args.label_fontsize,
                tick_fontsize=args.tick_fontsize,
                legend_fontsize=args.legend_fontsize,
                legend_ncol=args.legend_ncol,
                legend_y=args.legend_y,
                tight_left=args.tight_left,
                tight_bottom=args.tight_bottom,
                tight_right=args.tight_right,
                tight_top=args.tight_top,
            )

    if args.split_by_source:
        for source in ("youtube", "x"):
            source_items = [item for item in found if item[0] == source]
            if not source_items:
                continue
            source_out = (
                args.output_dir
                / f"{args.output_prefix}_{source}_{stem}_montage.png"
            )
            source_title = None if args.no_title else f"{args.title} ({source})"
            if args.render_mode == "image":
                source_cols = min(args.cols, len(source_items))
                make_montage(
                    items=source_items,
                    output_path=source_out,
                    cols=source_cols,
                    rows=args.rows,
                    cell_width=args.cell_width,
                    cell_height=args.cell_height,
                    dpi=args.dpi,
                    title=source_title,
                    fig_width_cm=args.fig_width_cm,
                    fig_height_cm=args.fig_height_cm,
                    tight_left=args.tight_left,
                    tight_bottom=args.tight_bottom,
                    tight_right=args.tight_right,
                    tight_top=args.tight_top,
                )
            else:
                make_curve_grid(
                    items=source_items,
                    output_path=source_out,
                    cols=args.cols,
                    rows=args.rows,
                    cell_width=args.cell_width,
                    cell_height=args.cell_height,
                    dpi=args.dpi,
                    title=source_title,
                    show_source_prefix=False,
                    trim_flat_tail_days=args.trim_flat_tail_days,
                    trim_flat_tail_eps=args.trim_flat_tail_eps,
                    x_max_ticks=args.x_max_ticks,
                    fig_width_cm=args.fig_width_cm,
                    fig_height_cm=args.fig_height_cm,
                    label_fontsize=args.label_fontsize,
                    tick_fontsize=args.tick_fontsize,
                    legend_fontsize=args.legend_fontsize,
                    legend_ncol=args.legend_ncol,
                    legend_y=args.legend_y,
                    tight_left=args.tight_left,
                    tight_bottom=args.tight_bottom,
                    tight_right=args.tight_right,
                    tight_top=args.tight_top,
                )

    manifest_out = args.output_dir / f"{args.output_prefix}_{stem}_manifest.csv"
    write_manifest(manifest_out, found, missing, artifact_name)

    if not args.source_only:
        print(f"Saved montage: {combined_out}")
    if args.split_by_source:
        print("Saved source montages under:", args.output_dir)
    print(f"Saved manifest: {manifest_out}")
    print(f"Found: {len(found)} | Missing: {len(missing)}")
    if missing:
        preview = ", ".join([f"{s}/{k}" for s, k in missing[:8]])
        suffix = " ..." if len(missing) > 8 else ""
        print(f"Missing entries: {preview}{suffix}")


if __name__ == "__main__":
    main()
