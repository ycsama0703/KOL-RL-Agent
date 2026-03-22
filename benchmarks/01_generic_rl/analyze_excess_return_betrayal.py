#!/usr/bin/env python3
"""Experiment 2: link between excess return and betrayal probability.

Question:
When a method beats baseline on an event (excess_return > 0), does betrayal
probability become significantly higher than its average / non-excess level?
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_METHOD_ORDER = ["KICL", "AWAC", "IQL", "BC", "CQL", "TD3BC"]
METHOD_COLORS = {
    "KICL": "#F39C12",
    "AWAC": "#17BECF",
    "IQL": "#2CA02C",
    "BC": "#8C564B",
    "CQL": "#D62728",
    "TD3BC": "#9467BD",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest-json",
        default="benchmarks/compare/meta/compare_manifest.json",
        help="Compare manifest containing method roots.",
    )
    p.add_argument(
        "--detailed-csv",
        default="benchmarks/compare/meta/selected20_all_methods_vs_baseline_detailed.csv",
        help="Selected subset rows (defines source/kol universe).",
    )
    p.add_argument(
        "--output-dir",
        default="benchmarks/compare/analysis_excess_return_betrayal",
        help="Output directory.",
    )
    p.add_argument(
        "--method-order",
        nargs="*",
        default=DEFAULT_METHOD_ORDER,
        help="Method order for outputs and plots.",
    )
    p.add_argument(
        "--pair-mode",
        choices=["intersection", "union"],
        default="intersection",
        help="Use source/kol pair intersection across methods, or union with missing skips.",
    )
    p.add_argument("--entry-threshold", type=float, default=0.02)
    p.add_argument("--action-threshold", type=float, default=0.02)
    p.add_argument(
        "--condition-mode",
        choices=["profit_event", "excess_vs_baseline_proxy"],
        default="profit_event",
        help=(
            "profit_event: condition is event_return > 0 (recommended, less mechanically tied to deviation); "
            "excess_vs_baseline_proxy: condition is (policy_action-baseline_action)*reward > 0."
        ),
    )
    p.add_argument(
        "--dev-threshold",
        type=float,
        default=0.20,
        help="Material deviation threshold on normalized dev for betrayal_any.",
    )
    p.add_argument(
        "--excess-eps",
        type=float,
        default=1e-12,
        help="Positive threshold for excess return condition.",
    )
    p.add_argument("--bootstrap-iters", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dpi", type=int, default=320)
    return p.parse_args()


def split_run_name(run_name: str) -> Tuple[str, str]:
    m = re.match(r"(.+)_([0-9]{8}_[0-9]{6})$", run_name)
    if not m:
        return run_name, ""
    return m.group(1), m.group(2)


def discover_latest_runs(root: Path) -> Dict[Tuple[str, str], Path]:
    out: Dict[Tuple[str, str], Tuple[str, Path]] = {}
    if not root.exists():
        return {}
    for source_dir in [p for p in root.iterdir() if p.is_dir()]:
        source = source_dir.name
        for run_dir in [p for p in source_dir.iterdir() if p.is_dir()]:
            kol, ts = split_run_name(run_dir.name)
            key = (source, kol)
            prev = out.get(key)
            if prev is None or ts > prev[0]:
                out[key] = (ts, run_dir)
    return {k: v[1] for k, v in out.items()}


def _event_rows_from_positions(
    df: pd.DataFrame,
    entry_threshold: float,
    action_threshold: float,
    dev_threshold: float,
    excess_eps: float,
    condition_mode: str,
) -> pd.DataFrame:
    req = {"reward", "baseline_action", "policy_action"}
    if not req.issubset(df.columns):
        return pd.DataFrame()

    out = df.copy()
    b = pd.to_numeric(out["baseline_action"], errors="coerce").fillna(0.0)
    p = pd.to_numeric(out["policy_action"], errors="coerce").fillna(0.0)
    r = pd.to_numeric(out["reward"], errors="coerce").fillna(0.0)
    if "weight" in out.columns:
        w = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
        event_return = w * r
    else:
        event_return = p * r

    eps = 1e-8
    has_signal = b.abs() >= float(entry_threshold)
    no_signal = ~has_signal
    reversal = has_signal & ((b * p) < 0.0)

    dev = np.zeros(len(out), dtype=float)
    idx_sig = has_signal.to_numpy()
    dev[idx_sig] = np.abs((p - b).to_numpy()[idx_sig]) / (np.abs(b.to_numpy()[idx_sig]) + eps)

    entry_v = np.zeros(len(out), dtype=float)
    idx_nosig = no_signal.to_numpy()
    denom = max(1.0 - float(action_threshold), eps)
    entry_v[idx_nosig] = np.maximum(0.0, np.abs(p.to_numpy()[idx_nosig]) - float(action_threshold)) / denom

    excess_return_proxy = (p - b) * r
    if condition_mode == "profit_event":
        cond_positive = event_return > float(excess_eps)
    else:
        cond_positive = excess_return_proxy > float(excess_eps)

    dev_flag = dev >= float(dev_threshold)
    entry_flag = entry_v > 0.0
    betrayal_any = reversal.to_numpy() | entry_flag | dev_flag

    out = pd.DataFrame(
        {
            "condition_positive": cond_positive.astype(int),
            "event_return": event_return.astype(float),
            "excess_return_proxy": excess_return_proxy.astype(float),
            "betrayal_any": betrayal_any.astype(int),
            "reversal_flag": reversal.astype(int),
            "entry_flag": entry_flag.astype(int),
            "dev_flag": dev_flag.astype(int),
        }
    )
    return out


def _mean_or_nan(x: Iterable[float]) -> float:
    arr = np.asarray(list(x), dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def _bootstrap_ci(
    values: np.ndarray, iters: int, seed: int, q_lo: float = 0.025, q_hi: float = 0.975
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = np.empty(iters, dtype=float)
    for i in range(iters):
        idx = rng.integers(0, n, size=n)
        boots[i] = np.mean(values[idx])
    return float(np.quantile(boots, q_lo)), float(np.quantile(boots, q_hi))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.manifest_json).read_text(encoding="utf-8"))
    method_roots = {k: Path(v) for k, v in manifest["methods"].items() if k in args.method_order}
    method_order = [m for m in args.method_order if m in method_roots]
    if not method_order:
        raise RuntimeError("No methods found from manifest/method-order intersection.")

    detailed = pd.read_csv(args.detailed_csv)
    universe_pairs = set(
        map(tuple, detailed[["source", "kol"]].drop_duplicates().itertuples(index=False, name=None))
    )
    if not universe_pairs:
        raise RuntimeError("No source/kol pairs found in detailed CSV.")

    latest_by_method = {m: discover_latest_runs(root) for m, root in method_roots.items()}
    pairs_by_method = {m: set(latest.keys()) & universe_pairs for m, latest in latest_by_method.items()}

    if args.pair_mode == "intersection":
        common_pairs = set.intersection(*(pairs_by_method[m] for m in method_order))
    else:
        common_pairs = set.union(*(pairs_by_method[m] for m in method_order))

    if not common_pairs:
        raise RuntimeError("No common source/kol pairs found for analysis.")

    # event-level rows
    all_rows = []
    coverage_rows = []
    for m in method_order:
        latest = latest_by_method[m]
        method_pairs = sorted(common_pairs if args.pair_mode == "intersection" else pairs_by_method[m])
        found = 0
        row_count = 0
        for src, kol in method_pairs:
            run_dir = latest.get((src, kol))
            if run_dir is None:
                continue
            pos = run_dir / "event" / "positions_test.csv"
            if not pos.exists():
                continue
            df = pd.read_csv(pos)
            ev = _event_rows_from_positions(
                df,
                entry_threshold=args.entry_threshold,
                action_threshold=args.action_threshold,
                dev_threshold=args.dev_threshold,
                excess_eps=args.excess_eps,
                condition_mode=args.condition_mode,
            )
            if ev.empty:
                continue
            ev["method"] = m
            ev["source"] = src
            ev["kol"] = kol
            all_rows.append(ev)
            found += 1
            row_count += len(ev)

        coverage_rows.append(
            {
                "method": m,
                "pair_mode": args.pair_mode,
                "pairs_target": len(method_pairs),
                "pairs_found": found,
                "event_rows": row_count,
            }
        )

    if not all_rows:
        raise RuntimeError("No event rows loaded.")
    events = pd.concat(all_rows, ignore_index=True)
    pd.DataFrame(coverage_rows).to_csv(out_dir / "coverage_summary.csv", index=False)

    # pooled stats by method/source
    pooled_rows = []
    by_kol_rows = []
    for (src, m), g in events.groupby(["source", "method"], sort=False):
        e = g["condition_positive"].to_numpy(dtype=int)
        b = g["betrayal_any"].to_numpy(dtype=int)

        n = len(g)
        n_e = int(e.sum())
        n_ne = int((1 - e).sum())

        p_b = float(b.mean()) if n > 0 else float("nan")
        p_b_e = float(b[e == 1].mean()) if n_e > 0 else float("nan")
        p_b_ne = float(b[e == 0].mean()) if n_ne > 0 else float("nan")
        uplift_avg = p_b_e - p_b if np.isfinite(p_b_e) and np.isfinite(p_b) else float("nan")
        uplift_ne = p_b_e - p_b_ne if np.isfinite(p_b_e) and np.isfinite(p_b_ne) else float("nan")
        rr_ne = p_b_e / p_b_ne if np.isfinite(p_b_e) and np.isfinite(p_b_ne) and p_b_ne > 0 else float("nan")

        pooled_rows.append(
            {
                "source": src,
                "method": m,
                "n_events": n,
                "n_excess_pos": n_e,
                "n_excess_nonpos": n_ne,
                "p_betrayal": p_b,
                "p_betrayal_given_excess_pos": p_b_e,
                "p_betrayal_given_excess_nonpos": p_b_ne,
                "uplift_vs_avg": uplift_avg,
                "uplift_vs_nonexcess": uplift_ne,
                "risk_ratio_excess_vs_nonexcess": rr_ne,
            }
        )

        for kol, kg in g.groupby("kol"):
            ek = kg["condition_positive"].to_numpy(dtype=int)
            bk = kg["betrayal_any"].to_numpy(dtype=int)
            n_ek = int(ek.sum())
            n_nek = int((1 - ek).sum())
            p_bk = float(bk.mean()) if len(kg) > 0 else float("nan")
            p_bk_e = float(bk[ek == 1].mean()) if n_ek > 0 else float("nan")
            p_bk_ne = float(bk[ek == 0].mean()) if n_nek > 0 else float("nan")
            by_kol_rows.append(
                {
                    "source": src,
                    "method": m,
                    "kol": kol,
                    "n_events": len(kg),
                    "n_excess_pos": n_ek,
                    "n_excess_nonpos": n_nek,
                    "p_betrayal": p_bk,
                    "p_betrayal_given_excess_pos": p_bk_e,
                    "p_betrayal_given_excess_nonpos": p_bk_ne,
                    "uplift_vs_avg": p_bk_e - p_bk
                    if np.isfinite(p_bk_e) and np.isfinite(p_bk)
                    else float("nan"),
                    "uplift_vs_nonexcess": p_bk_e - p_bk_ne
                    if np.isfinite(p_bk_e) and np.isfinite(p_bk_ne)
                    else float("nan"),
                }
            )

    pooled_df = pd.DataFrame(pooled_rows)
    by_kol_df = pd.DataFrame(by_kol_rows)
    pooled_df.to_csv(out_dir / "excess_return_betrayal_pooled.csv", index=False)
    by_kol_df.to_csv(out_dir / "excess_return_betrayal_by_kol.csv", index=False)

    # bootstrap CI over KOL-level uplifts
    ci_rows = []
    for (src, m), g in by_kol_df.groupby(["source", "method"], sort=False):
        d1 = g["uplift_vs_avg"].to_numpy(dtype=float)
        d2 = g["uplift_vs_nonexcess"].to_numpy(dtype=float)
        d1 = d1[np.isfinite(d1)]
        d2 = d2[np.isfinite(d2)]
        ci1 = _bootstrap_ci(d1, args.bootstrap_iters, seed=args.seed + 101)
        ci2 = _bootstrap_ci(d2, args.bootstrap_iters, seed=args.seed + 313)
        mu1 = _mean_or_nan(d1)
        mu2 = _mean_or_nan(d2)
        ci_rows.append(
            {
                "source": src,
                "method": m,
                "n_kols_used_uplift_vs_avg": len(d1),
                "uplift_vs_avg_mean_kol": mu1,
                "uplift_vs_avg_ci_low": ci1[0],
                "uplift_vs_avg_ci_high": ci1[1],
                "uplift_vs_avg_ci_excludes_zero": bool(np.isfinite(ci1[0]) and np.isfinite(ci1[1]) and (ci1[0] > 0 or ci1[1] < 0)),
                "n_kols_used_uplift_vs_nonexcess": len(d2),
                "uplift_vs_nonexcess_mean_kol": mu2,
                "uplift_vs_nonexcess_ci_low": ci2[0],
                "uplift_vs_nonexcess_ci_high": ci2[1],
                "uplift_vs_nonexcess_ci_excludes_zero": bool(np.isfinite(ci2[0]) and np.isfinite(ci2[1]) and (ci2[0] > 0 or ci2[1] < 0)),
            }
        )
    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(out_dir / "excess_return_betrayal_bootstrap_ci.csv", index=False)

    # Plot: overall vs excess-pos betrayal probability
    sources = [s for s in ["x", "youtube"] if s in set(pooled_df["source"])]
    fig, axes = plt.subplots(1, len(sources), figsize=(8.6, 3.6), squeeze=False)
    axes = axes[0]
    for ax, src in zip(axes, sources):
        sdf = pooled_df[pooled_df["source"] == src].copy()
        methods_present = [m for m in method_order if m in set(sdf["method"])]
        y_pos = np.arange(len(methods_present))

        for i, m in enumerate(methods_present):
            r = sdf[sdf["method"] == m].iloc[0]
            x1 = float(r["p_betrayal"])
            x2 = float(r["p_betrayal_given_excess_pos"])
            color = METHOD_COLORS.get(m, "#333333")
            if np.isfinite(x1) and np.isfinite(x2):
                ax.plot([x1, x2], [i, i], color=color, linewidth=1.4, alpha=0.9)
                ax.scatter([x1], [i], s=35, color="#666666", zorder=3, marker="o")
                ax.scatter([x2], [i], s=62, color=color, zorder=4, marker="D", edgecolors="white", linewidths=0.5)
                ax.text(x2 + 0.01, i, m, va="center", ha="left", fontsize=8.0, color="#222222")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(methods_present, fontsize=8.2)
        ax.set_xlabel("Betrayal probability", fontsize=9.0)
        ax.set_title("X" if src == "x" else "YouTube", fontsize=10.5)
        ax.grid(True, axis="x", linestyle="--", alpha=0.25, linewidth=0.6)
        ax.set_xlim(0.0, 1.0)
        if src != sources[0]:
            ax.tick_params(labelleft=False)

    fig.supylabel("Method", fontsize=9.2)
    cond_txt = "event_return > 0" if args.condition_mode == "profit_event" else "(policy-baseline)*reward > 0"
    fig.supxlabel(f"Circle: P(B) overall, Diamond: P(B | {cond_txt})", fontsize=9.0, y=0.02)
    fig.tight_layout(rect=(0.02, 0.05, 1, 1), w_pad=1.1)
    fig.savefig(out_dir / "excess_return_betrayal_probability_shift.png", dpi=args.dpi)
    fig.savefig(out_dir / "excess_return_betrayal_probability_shift.pdf")
    plt.close(fig)

    # markdown summary
    md_lines = []
    md_lines.append("# Excess-Return vs Betrayal (Experiment 2)")
    md_lines.append("")
    md_lines.append(f"- Pair mode: `{args.pair_mode}`")
    md_lines.append(f"- Condition mode: `{args.condition_mode}`")
    md_lines.append(f"- Universe pairs in detailed CSV: `{len(universe_pairs)}`")
    md_lines.append(f"- Common pairs actually used: `{len(common_pairs)}`")
    md_lines.append("")
    md_lines.append("Definition:")
    md_lines.append("- `event_return = weight * reward` (or `policy_action * reward` if `weight` is absent)")
    md_lines.append("- `excess_return_proxy = (policy_action - baseline_action) * reward`")
    if args.condition_mode == "profit_event":
        md_lines.append("- Condition event: `event_return > 0`")
    else:
        md_lines.append("- Condition event: `excess_return_proxy > 0`")
    md_lines.append(
        f"- `betrayal_any = reversal OR entry_violation OR (dev >= {args.dev_threshold:.3f})`"
    )
    md_lines.append("")
    md_lines.append("Key files:")
    md_lines.append("- `coverage_summary.csv`")
    md_lines.append("- `excess_return_betrayal_pooled.csv`")
    md_lines.append("- `excess_return_betrayal_bootstrap_ci.csv`")
    md_lines.append("- `excess_return_betrayal_probability_shift.png`")
    (out_dir / "README.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Saved: {out_dir / 'coverage_summary.csv'}")
    print(f"Saved: {out_dir / 'excess_return_betrayal_pooled.csv'}")
    print(f"Saved: {out_dir / 'excess_return_betrayal_by_kol.csv'}")
    print(f"Saved: {out_dir / 'excess_return_betrayal_bootstrap_ci.csv'}")
    print(f"Saved: {out_dir / 'excess_return_betrayal_probability_shift.png'}")
    print(f"Saved: {out_dir / 'excess_return_betrayal_probability_shift.pdf'}")
    print(f"Saved: {out_dir / 'README.md'}")


if __name__ == "__main__":
    main()
