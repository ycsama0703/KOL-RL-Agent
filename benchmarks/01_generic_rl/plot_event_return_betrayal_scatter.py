"""Event-level return vs betrayal scatter from test outputs.

Each sample point is one row in event/positions_test.csv (date, ticker).

Definitions (per sample):
- event_return = weight * reward
- has_signal = |baseline_action| >= entry_threshold
- reversal = 1[has_signal and baseline_action * policy_action < 0]
- dev = |policy_action - baseline_action| / (|baseline_action| + eps), only when has_signal else 0
- entry_violation = max(0, |policy_action| - action_threshold) / (1 - action_threshold), only when no-signal else 0
- betrayal_index = dev + reversal + entry_violation
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd


DEFAULT_KOLS = [
    "Ale_s_World_of_Stocks",
    "Invest_with_Henry",
    "Dividend_Data",
    "The_Maverick_of_Wall_Street",
    "MarketBeat",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot event-level return vs betrayal scatter.")
    p.add_argument("--ours-root", default="outputs/multisource_test_mainline")
    p.add_argument("--ours-name", default="KICL")
    p.add_argument("--method", action="append", default=[], help="NAME=PATH, repeatable")
    p.add_argument("--source", default="youtube", help="Data source folder to use (youtube/x).")
    p.add_argument("--kols", default=",".join(DEFAULT_KOLS), help="Comma-separated KOL list.")
    p.add_argument(
        "--output-prefix",
        default="benchmarks/compare/youtube_event_tradeoff_selected5",
        help="Output prefix for png/pdf/csv files.",
    )
    p.add_argument("--entry-threshold", type=float, default=0.02)
    p.add_argument("--action-threshold", type=float, default=0.02)
    p.add_argument("--sample-cap-per-method", type=int, default=8000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--plot-active-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only plot active events (has_signal or entry_violation>0).",
    )
    p.add_argument(
        "--plot-betrayal-mode",
        choices=["raw", "hard_soft"],
        default="hard_soft",
        help=(
            "raw: betrayal_index = dev + reversal + entry_violation; "
            "hard_soft: 2*reversal + 2*entry_violation + 0.1*clip(dev,0,5)."
        ),
    )
    p.add_argument(
        "--plot-x-mode",
        choices=["betrayal", "intent_consistency"],
        default="intent_consistency",
        help=(
            "betrayal: lower is better; "
            "intent_consistency: x=1/(1+betrayal_plot), higher is better."
        ),
    )
    p.add_argument(
        "--plot-y-transform",
        choices=["raw_bps", "signed_log_bps"],
        default="signed_log_bps",
        help="Y-axis transform for visualization only.",
    )
    p.add_argument(
        "--mean-y-mode",
        choices=["all", "nonzero"],
        default="nonzero",
        help="How to compute method mean marker on Y: all events or nonzero-return events only.",
    )
    p.add_argument(
        "--render-mode",
        choices=["ellipse", "points", "both"],
        default="ellipse",
        help="How to render method clusters: ellipse-only, points-only, or both.",
    )
    p.add_argument(
        "--ellipse-trim-quantile",
        type=float,
        default=0.95,
        help="Axis-wise trim quantile used before fitting covariance ellipse.",
    )
    p.add_argument(
        "--ellipse-nstd",
        type=float,
        default=1.8,
        help="Std radius for the filled cluster ellipse.",
    )
    p.add_argument(
        "--ellipse-alpha",
        type=float,
        default=0.16,
        help="Opacity for cluster ellipse fill.",
    )
    p.add_argument(
        "--ellipse-center-mode",
        choices=["fitted_mean", "method_mean"],
        default="method_mean",
        help="Ellipse center source: fitted_mean (trimmed sample mean) or method_mean (mean marker).",
    )
    p.add_argument(
        "--focus-on-means",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, axis limits are auto-zoomed around method mean markers.",
    )
    p.add_argument(
        "--mean-focus-pad-ratio",
        type=float,
        default=0.35,
        help="Padding ratio around mean marker range when --focus-on-means is enabled.",
    )
    p.add_argument(
        "--x-plot-quantile",
        type=float,
        default=0.99,
        help="Upper quantile of betrayal_index to show in plot (robust zoom).",
    )
    p.add_argument(
        "--y-abs-plot-quantile",
        type=float,
        default=0.995,
        help="Upper quantile of |event_return| to show in plot (robust zoom).",
    )
    return p.parse_args()


def parse_method_args(extra: List[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for raw in extra:
        if "=" not in raw:
            raise ValueError(f"Invalid --method: {raw}; expected NAME=PATH")
        k, v = raw.split("=", 1)
        out[k.strip()] = Path(v.strip())
    return out


def split_run_name(run_name: str) -> Tuple[str, str]:
    m = re.match(r"(.+)_([0-9]{8}_[0-9]{6})$", run_name)
    if not m:
        return run_name, ""
    return m.group(1), m.group(2)


def discover_latest_runs(root: Path) -> Dict[Tuple[str, str], Path]:
    out: Dict[Tuple[str, str], Tuple[str, Path]] = {}
    if not root.exists():
        return {}
    for source_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        source = source_dir.name
        for run_dir in sorted([p for p in source_dir.iterdir() if p.is_dir()]):
            kol, ts = split_run_name(run_dir.name)
            key = (source, kol)
            prev = out.get(key)
            if prev is None or ts > prev[0]:
                out[key] = (ts, run_dir)
    return {k: v[1] for k, v in out.items()}


def compute_event_points(
    df: pd.DataFrame,
    entry_threshold: float,
    action_threshold: float,
) -> pd.DataFrame:
    req = {"reward", "weight", "baseline_action", "policy_action", "date", "ticker"}
    if not req.issubset(df.columns):
        return pd.DataFrame()

    out = df.copy()
    b = pd.to_numeric(out["baseline_action"], errors="coerce").fillna(0.0)
    p = pd.to_numeric(out["policy_action"], errors="coerce").fillna(0.0)
    w = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    r = pd.to_numeric(out["reward"], errors="coerce").fillna(0.0)

    eps = 1e-8
    has_signal = b.abs() >= float(entry_threshold)
    no_signal = ~has_signal
    reversal = (has_signal & ((b * p) < 0.0)).astype(float)

    dev = np.zeros(len(out), dtype=float)
    idx_sig = has_signal.to_numpy()
    dev[idx_sig] = np.abs((p - b).to_numpy()[idx_sig]) / (np.abs(b.to_numpy()[idx_sig]) + eps)

    entry_v = np.zeros(len(out), dtype=float)
    idx_nosig = no_signal.to_numpy()
    denom = max(1.0 - float(action_threshold), eps)
    entry_v[idx_nosig] = np.maximum(0.0, np.abs(p.to_numpy()[idx_nosig]) - float(action_threshold)) / denom

    out["event_return"] = w * r
    out["betrayal_index"] = dev + reversal + entry_v
    out["has_signal"] = has_signal.astype(int)
    out["reversal"] = reversal
    out["dev"] = dev
    out["entry_violation"] = entry_v
    return out[
        [
            "date",
            "ticker",
            "event_return",
            "betrayal_index",
            "has_signal",
            "reversal",
            "dev",
            "entry_violation",
        ]
    ]


def palette() -> Dict[str, str]:
    return {
        "KICL": "#ff7f0e",
        "BC": "#8c564b",
        "IQL": "#2ca02c",
        "CQL": "#d62728",
        "TD3BC": "#9467bd",
        "AWAC": "#17becf",
    }


def draw_cluster_ellipse(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    n_std: float = 1.8,
    alpha: float = 0.16,
    center: tuple[float, float] | None = None,
) -> None:
    if len(x) < 20:
        return
    cov = np.cov(np.vstack([x, y]))
    if cov.shape != (2, 2) or not np.isfinite(cov).all():
        return
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    if np.any(eigvals <= 0):
        return

    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2.0 * n_std * np.sqrt(eigvals)
    if center is None:
        center = (float(np.mean(x)), float(np.mean(y)))
    ell = Ellipse(
        xy=center,
        width=float(width),
        height=float(height),
        angle=float(angle),
        facecolor=color,
        edgecolor=color,
        linewidth=1.0,
        alpha=alpha,
        zorder=2,
    )
    ax.add_patch(ell)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    methods: Dict[str, Path] = {args.ours_name: Path(args.ours_root)}
    methods.update(parse_method_args(args.method))
    method_order = list(methods.keys())

    kols = [x.strip() for x in args.kols.split(",") if x.strip()]
    all_rows: List[pd.DataFrame] = []

    for method, root in methods.items():
        runs = discover_latest_runs(root)
        for kol in kols:
            key = (args.source, kol)
            run_dir = runs.get(key)
            if run_dir is None:
                continue
            pos_path = run_dir / "event" / "positions_test.csv"
            if not pos_path.exists():
                continue
            raw = pd.read_csv(pos_path)
            pts = compute_event_points(
                raw,
                entry_threshold=args.entry_threshold,
                action_threshold=args.action_threshold,
            )
            if pts.empty:
                continue
            pts["method"] = method
            pts["kol"] = kol
            pts["run_name"] = run_dir.name
            all_rows.append(pts)

    if not all_rows:
        raise SystemExit("No valid event points found. Check roots/source/kols.")

    points = pd.concat(all_rows, ignore_index=True)
    points["date"] = pd.to_datetime(points["date"], errors="coerce")
    points = points.dropna(subset=["event_return", "betrayal_index"])

    # Derived plotting features.
    points["event_return_bps"] = points["event_return"] * 1e4
    points["is_active"] = (points["has_signal"] > 0.5) | (points["entry_violation"] > 0.0)
    if args.plot_betrayal_mode == "raw":
        points["betrayal_plot"] = points["betrayal_index"]
    else:
        points["betrayal_plot"] = (
            2.0 * points["reversal"].clip(lower=0.0, upper=1.0)
            + 2.0 * points["entry_violation"].clip(lower=0.0, upper=1.0)
            + 0.1 * points["dev"].clip(lower=0.0, upper=5.0)
        )
    if args.plot_y_transform == "raw_bps":
        points["event_return_plot"] = points["event_return_bps"]
        y_label = "Event Return Contribution (bps, weight × reward)"
    else:
        bps = points["event_return_bps"]
        points["event_return_plot"] = np.sign(bps) * np.log1p(np.abs(bps))
        y_label = "Signed log Event Return (bps)"

    # Optional downsample for readability (after derived fields exist).
    sampled_chunks = []
    for method, g in points.groupby("method"):
        if len(g) > args.sample_cap_per_method:
            idx = rng.choice(g.index.to_numpy(), size=args.sample_cap_per_method, replace=False)
            sampled_chunks.append(g.loc[idx])
        else:
            sampled_chunks.append(g)
    sampled = pd.concat(sampled_chunks, ignore_index=True)

    if args.plot_active_only:
        sampled = sampled[sampled["is_active"] == True].copy()

    # Robust display bounds to avoid axis collapse by extreme tails.
    q_base = points[points["is_active"] == True].copy() if args.plot_active_only else points.copy()
    if args.plot_x_mode == "intent_consistency":
        q_base["x_plot"] = 1.0 / (1.0 + q_base["betrayal_plot"].clip(lower=0.0))
        sampled["x_plot"] = 1.0 / (1.0 + sampled["betrayal_plot"].clip(lower=0.0))
        plot_x_lo = float(q_base["x_plot"].quantile(1.0 - args.x_plot_quantile))
        plot_x_hi = 1.0
    else:
        q_base["x_plot"] = q_base["betrayal_plot"]
        sampled["x_plot"] = sampled["betrayal_plot"]
        plot_x_lo = 0.0
        plot_x_hi = float(q_base["x_plot"].quantile(args.x_plot_quantile))

    y_cap = float(q_base["event_return_plot"].abs().quantile(args.y_abs_plot_quantile))
    plot_x_hi = max(plot_x_hi, 1e-6)
    y_cap = max(y_cap, 1e-8)

    in_view = (
        (sampled["x_plot"] >= plot_x_lo)
        & (sampled["x_plot"] <= plot_x_hi)
        & (sampled["event_return_plot"].abs() <= y_cap)
    )
    sampled_view = sampled[in_view].copy()
    dropped = int((~in_view).sum())
    kept = int(in_view.sum())

    summary_rows = []
    for method, g in q_base.groupby("method"):
        nz = g["event_return"].abs() > 1e-12
        g_y = g[nz] if args.mean_y_mode == "nonzero" and nz.any() else g
        summary_rows.append(
            {
                "method": method,
                "mean_event_return": float(g["event_return"].mean()),
                "mean_betrayal_index": float(g["betrayal_index"].mean()),
                "mean_betrayal_plot": float(g["betrayal_plot"].mean()),
                "mean_event_return_plot": float(g_y["event_return_plot"].mean()),
                "mean_dev": float(g["dev"].mean()),
                "mean_entry_violation": float(g["entry_violation"].mean()),
                "mean_reversal": float(g["reversal"].mean()),
                "n_events": int(len(g)),
                "n_nonzero_events": int(nz.sum()),
                "nonzero_ratio": float(nz.mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)

    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.6, 6.4))
    ax.set_facecolor("#fbfcff")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.28)

    colors = palette()
    for method in method_order:
        g = sampled_view[sampled_view["method"] == method]
        if g.empty:
            continue
        c = colors.get(method, "#4c4c4c")
        m = summary[summary["method"] == method]
        mean_x = None
        mean_y = None
        if not m.empty:
            mean_b = float(m["mean_betrayal_plot"].iloc[0])
            mean_x = (1.0 / (1.0 + max(mean_b, 0.0))) if args.plot_x_mode == "intent_consistency" else mean_b
            mean_y = float(m["mean_event_return_plot"].iloc[0])

        if args.render_mode in {"points", "both"}:
            ax.scatter(
                g["x_plot"],
                g["event_return_plot"],
                s=9,
                alpha=0.16,
                color=c,
                edgecolors="none",
                label=f"{method} events (view)",
                zorder=1,
            )
        if args.render_mode in {"ellipse", "both"}:
            gx = g["x_plot"].to_numpy(dtype=float)
            gy = g["event_return_plot"].to_numpy(dtype=float)
            if len(gx) >= 20:
                q = min(max(args.ellipse_trim_quantile, 0.6), 0.995)
                ex_lo, ex_hi = np.quantile(gx, 1.0 - q), np.quantile(gx, q)
                ey_lo, ey_hi = np.quantile(gy, 1.0 - q), np.quantile(gy, q)
                keep = (gx >= ex_lo) & (gx <= ex_hi) & (gy >= ey_lo) & (gy <= ey_hi)
                center_override = None
                if args.ellipse_center_mode == "method_mean" and mean_x is not None and mean_y is not None:
                    center_override = (float(mean_x), float(mean_y))
                draw_cluster_ellipse(
                    ax=ax,
                    x=gx[keep],
                    y=gy[keep],
                    color=c,
                    n_std=args.ellipse_nstd,
                    alpha=args.ellipse_alpha,
                    center=center_override,
                )
        if not m.empty:
            x = float(mean_x)
            y = float(mean_y)
            marker = "X" if method == args.ours_name else "o"
            size = 220 if method == args.ours_name else 130
            ax.scatter(
                [x],
                [y],
                s=size,
                marker=marker,
                color=c,
                edgecolors="black" if method == args.ours_name else "white",
                linewidths=1.0,
                zorder=6,
                label=f"{method} mean",
            )
            ax.annotate(
                method,
                (x, y),
                xytext=(6, 5),
                textcoords="offset points",
                fontsize=10,
                weight="bold" if method == args.ours_name else "normal",
            )

    if args.focus_on_means and not summary.empty:
        mean_x = []
        mean_y = []
        for _, row in summary.iterrows():
            mb = float(row["mean_betrayal_plot"])
            mx = (1.0 / (1.0 + max(mb, 0.0))) if args.plot_x_mode == "intent_consistency" else mb
            my = float(row["mean_event_return_plot"])
            mean_x.append(mx)
            mean_y.append(my)
        xmn, xmx = float(np.min(mean_x)), float(np.max(mean_x))
        ymn, ymx = float(np.min(mean_y)), float(np.max(mean_y))
        xspan = max(xmx - xmn, 1e-6)
        yspan = max(ymx - ymn, 1e-6)
        pad = max(args.mean_focus_pad_ratio, 0.05)
        zx_lo = xmn - pad * xspan
        zx_hi = xmx + pad * xspan
        zy_lo = ymn - pad * yspan
        zy_hi = ymx + pad * yspan
        if args.plot_x_mode == "intent_consistency":
            zx_lo = max(0.0, zx_lo)
            zx_hi = min(1.0, zx_hi)
        else:
            zx_lo = max(0.0, zx_lo)
        # Keep reference line y=0 visible for interpretation.
        zy_lo = min(zy_lo, 0.0)
        zy_hi = max(zy_hi, 0.0)
        ax.set_xlim(left=zx_lo, right=zx_hi)
        ax.set_ylim(bottom=zy_lo, top=zy_hi)
    else:
        ax.set_xlim(left=plot_x_lo, right=plot_x_hi)
        ax.set_ylim(bottom=-y_cap, top=y_cap)

    ax.axhline(0.0, color="#666666", linestyle=":", linewidth=1.0, alpha=0.7)
    if args.plot_x_mode == "intent_consistency":
        ax.set_xlabel(
            "Event Intent Consistency = 1/(1 + betrayal_plot) (higher is better)"
        )
    else:
        ax.set_xlabel(
            "Event Betrayal Index (plot mode: "
            + ("raw" if args.plot_betrayal_mode == "raw" else "hard_soft")
            + ", lower is better)"
        )
    ax.set_ylabel(y_label)
    ax.set_title(f"Event-Level Return vs Betrayal (Robust View, {args.source})")
    ax.text(
        0.01,
        0.98,
        (
            f"view cap: x∈[{plot_x_lo:.3f},{plot_x_hi:.3f}], |y|≤{y_cap:.2f}\n"
            f"active_only={args.plot_active_only}, y={args.plot_y_transform}, mean_y={args.mean_y_mode}, render={args.render_mode}, ellipse_center={args.ellipse_center_mode}, focus_means={args.focus_on_means}\n"
            f"plotted: {kept:,}/{len(sampled):,} sampled events"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", alpha=0.85),
    )
    ax.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.92, ncol=2)
    fig.tight_layout()

    fig.savefig(out_prefix.with_suffix(".png"), dpi=230)
    fig.savefig(out_prefix.with_suffix(".pdf"))
    plt.close(fig)

    points.to_csv(out_prefix.with_name(out_prefix.name + "_points.csv"), index=False)
    sampled.to_csv(out_prefix.with_name(out_prefix.name + "_sampled_points.csv"), index=False)
    sampled_view.to_csv(out_prefix.with_name(out_prefix.name + "_sampled_points_in_view.csv"), index=False)
    summary.to_csv(out_prefix.with_name(out_prefix.name + "_method_mean.csv"), index=False)

    print(f"Saved plot: {out_prefix.with_suffix('.png')}")
    print(f"Saved summary: {out_prefix.with_name(out_prefix.name + '_method_mean.csv')}")
    print(f"Total events: {len(points):,}; sampled events: {len(sampled):,}; in-view: {kept:,}; clipped: {dropped:,}")


if __name__ == "__main__":
    main()
