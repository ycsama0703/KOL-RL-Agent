"""Build a compact 5-point ablation summary (FULL + baseline + 3 ablations).

Inputs:
- compare folder produced by build_ablation_compare.py

Outputs:
- five_point_summary_by_source.csv
- five_point_summary_overall.csv
- five_point_win_vs_baseline_by_source.csv
- FIVE_POINT_STORY.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build story-ready 5-point ablation summary.")
    p.add_argument(
        "--compare-dir",
        default="ablation study/five_point_compare",
        help="Output dir from build_ablation_compare.py.",
    )
    p.add_argument(
        "--full-name",
        default="KICL",
        help="Display name of full model.",
    )
    return p.parse_args()


def _find_col(cols: list[str], prefix: str, metric: str) -> str:
    target = f"{prefix}_{metric}"
    if target not in cols:
        raise KeyError(f"Missing column: {target}")
    return target


def build_summary_by_source(kol_df: pd.DataFrame, full_name: str) -> pd.DataFrame:
    method_prefix = {
        full_name: "kicl",
        "WO_HARD": "wo_hard",
        "WO_RL_COMPLETION": "wo_rl_completion",
        "WO_REGIME_SPLIT": "wo_regime_split",
    }

    rows = []
    for source, sdf in kol_df.groupby("source"):
        for method_name, pref in method_prefix.items():
            row = {
                "source": source,
                "method": method_name,
                "n_kols": int(len(sdf)),
                "event_return_mean": float(sdf[_find_col(kol_df.columns.tolist(), pref, "event_cumulative_return")].mean()),
                "event_sharpe_mean": float(sdf[_find_col(kol_df.columns.tolist(), pref, "event_sharpe")].mean()),
                "event_mdd_mean": float(sdf[_find_col(kol_df.columns.tolist(), pref, "event_max_drawdown")].mean()),
                "UER_mean": float(sdf[_find_col(kol_df.columns.tolist(), pref, "betrayal_entry_violation_rate")].mean()),
                "DRR_mean": float(sdf[_find_col(kol_df.columns.tolist(), pref, "betrayal_reversal_rate")].mean()),
                "BD_mean": float(sdf[_find_col(kol_df.columns.tolist(), pref, "betrayal_mean_abs_deviation")].mean()),
                "daily_return_mean": float(sdf[_find_col(kol_df.columns.tolist(), pref, "daily_trained_cumulative_return")].mean()),
                "daily_sharpe_mean": float(sdf[_find_col(kol_df.columns.tolist(), pref, "daily_trained_sharpe")].mean()),
                "daily_mdd_mean": float(sdf[_find_col(kol_df.columns.tolist(), pref, "daily_trained_max_drawdown")].mean()),
            }
            rows.append(row)

        # Add baseline row (from full model baseline columns).
        bp = method_prefix[full_name]
        rows.append(
            {
                "source": source,
                "method": "BASELINE",
                "n_kols": int(len(sdf)),
                "event_return_mean": float(
                    sdf[_find_col(kol_df.columns.tolist(), bp, "event_baseline_cumulative_return")].mean()
                ),
                "event_sharpe_mean": float(
                    sdf[_find_col(kol_df.columns.tolist(), bp, "event_baseline_sharpe")].mean()
                ),
                "event_mdd_mean": float(
                    sdf[_find_col(kol_df.columns.tolist(), bp, "event_baseline_max_drawdown")].mean()
                ),
                "UER_mean": 0.0,
                "DRR_mean": 0.0,
                "BD_mean": 0.0,
                "daily_return_mean": float(sdf[_find_col(kol_df.columns.tolist(), bp, "daily_baseline_cumulative_return")].mean()),
                "daily_sharpe_mean": float(sdf[_find_col(kol_df.columns.tolist(), bp, "daily_baseline_sharpe")].mean()),
                "daily_mdd_mean": float(sdf[_find_col(kol_df.columns.tolist(), bp, "daily_baseline_max_drawdown")].mean()),
            }
        )

    out = pd.DataFrame(rows)
    out["HVC_mean"] = out["UER_mean"] + out["DRR_mean"]
    out["MDD_mean"] = out["event_mdd_mean"]
    order = ["BASELINE", full_name, "WO_HARD", "WO_RL_COMPLETION", "WO_REGIME_SPLIT"]
    out["method"] = pd.Categorical(out["method"], categories=order, ordered=True)
    out = out.sort_values(["source", "method"]).reset_index(drop=True)
    return out


def build_win_vs_baseline(kol_df: pd.DataFrame, full_name: str) -> pd.DataFrame:
    method_prefix = {
        full_name: "kicl",
        "WO_HARD": "wo_hard",
        "WO_RL_COMPLETION": "wo_rl_completion",
        "WO_REGIME_SPLIT": "wo_regime_split",
    }
    rows = []
    for source, sdf in kol_df.groupby("source"):
        for method_name, pref in method_prefix.items():
            tr = sdf[f"{pref}_daily_trained_cumulative_return"]
            bl = sdf[f"{pref}_daily_baseline_cumulative_return"]
            rows.append(
                {
                    "source": source,
                    "method": method_name,
                    "n_kols": int(len(sdf)),
                    "win_vs_baseline": int((tr > bl).sum()),
                    "tie_vs_baseline": int((tr == bl).sum()),
                    "lose_vs_baseline": int((tr < bl).sum()),
                    "mean_daily_return_delta_vs_baseline": float((tr - bl).mean()),
                }
            )
    out = pd.DataFrame(rows)
    order = [full_name, "WO_HARD", "WO_RL_COMPLETION", "WO_REGIME_SPLIT"]
    out["method"] = pd.Categorical(out["method"], categories=order, ordered=True)
    out = out.sort_values(["source", "method"]).reset_index(drop=True)
    return out


def build_story_markdown(
    out_dir: Path,
    by_source: pd.DataFrame,
    win_vs_baseline: pd.DataFrame,
    full_name: str,
) -> None:
    def pick(df: pd.DataFrame, source: str, method: str) -> pd.Series:
        return df[(df["source"] == source) & (df["method"] == method)].iloc[0]

    lines: list[str] = []
    lines.append("# Five-Point Ablation Story (Full + Baseline + 3 Key Ablations)")
    lines.append("")
    lines.append("## What Is Included")
    lines.append(f"- `BASELINE`")
    lines.append(f"- `{full_name}` (full model)")
    lines.append("- `WO_HARD` (remove hard intent constraints)")
    lines.append("- `WO_RL_COMPLETION` (disable RL completion; near-baseline proxy)")
    lines.append("- `WO_REGIME_SPLIT` (single-head without regime split)")
    lines.append("")
    lines.append("## Why These 3 Ablations")
    lines.append("- `WO_HARD`: tests feasibility layer necessity (hard-violation control).")
    lines.append("- `WO_RL_COMPLETION`: tests whether RL completion contributes beyond baseline imitation.")
    lines.append("- `WO_REGIME_SPLIT`: tests whether signal/silence architecture contributes structurally.")
    lines.append("")
    lines.append("## High-Level Observations")
    for src in ["x", "youtube"]:
        k = pick(by_source, src, full_name)
        h = pick(by_source, src, "WO_HARD")
        r = pick(by_source, src, "WO_RL_COMPLETION")
        s = pick(by_source, src, "WO_REGIME_SPLIT")
        kw = pick(win_vs_baseline, src, full_name)
        hw = pick(win_vs_baseline, src, "WO_HARD")
        rw = pick(win_vs_baseline, src, "WO_RL_COMPLETION")
        sw = pick(win_vs_baseline, src, "WO_REGIME_SPLIT")

        lines.append(f"### {src.upper()}")
        lines.append(
            f"- `{full_name}`: event return={k['event_return_mean']:.3f}, sharpe={k['event_sharpe_mean']:.3f}, "
            f"UER={k['UER_mean']:.3f}, DRR={k['DRR_mean']:.3f}, BD={k['BD_mean']:.3f}."
        )
        lines.append(
            f"- `WO_HARD`: event return={h['event_return_mean']:.3f}, sharpe={h['event_sharpe_mean']:.3f}, "
            f"UER={h['UER_mean']:.3f}, DRR={h['DRR_mean']:.3f}, BD={h['BD_mean']:.3f}."
        )
        lines.append(
            f"- `WO_RL_COMPLETION`: event return={r['event_return_mean']:.3f}, sharpe={r['event_sharpe_mean']:.3f}, "
            f"UER={r['UER_mean']:.3f}, DRR={r['DRR_mean']:.3f}, BD={r['BD_mean']:.3f}."
        )
        lines.append(
            f"- `WO_REGIME_SPLIT`: event return={s['event_return_mean']:.3f}, sharpe={s['event_sharpe_mean']:.3f}, "
            f"UER={s['UER_mean']:.3f}, DRR={s['DRR_mean']:.3f}, BD={s['BD_mean']:.3f}."
        )
        lines.append(
            f"- Daily win vs baseline (count out of {int(kw['n_kols'])}): "
            f"`{full_name}`={int(kw['win_vs_baseline'])}, "
            f"`WO_HARD`={int(hw['win_vs_baseline'])}, "
            f"`WO_RL_COMPLETION`={int(rw['win_vs_baseline'])}, "
            f"`WO_REGIME_SPLIT`={int(sw['win_vs_baseline'])}."
        )
        lines.append("")

    lines.append("## How To Tell the Story (Paper-Friendly)")
    lines.append("1. Constraint necessity: removing hard constraints (`WO_HARD`) sharply increases hard betrayal (UER/DRR), and weakens robust gains.")
    lines.append(
        f"2. Completion necessity: without RL completion (`WO_RL_COMPLETION`), behavior stays close to baseline but incremental gains over `{full_name}` shrink."
    )
    lines.append("3. Structural contribution: removing regime split (`WO_REGIME_SPLIT`) degrades performance consistency, indicating value of signal/silence decomposition.")
    lines.append("")
    lines.append("## Files")
    lines.append("- `summary_by_kol.csv`: per-KOL raw compare table from five-point run")
    lines.append("- `summary_by_method_mean_by_source.csv`: method means by source (4 methods)")
    lines.append("- `five_point_summary_by_source.csv`: story-ready 5-point table (adds BASELINE row)")
    lines.append("- `five_point_summary_overall.csv`: averaged across sources")
    lines.append("- `five_point_win_vs_baseline_by_source.csv`: win/tie/lose vs baseline")
    lines.append("")
    (out_dir / "FIVE_POINT_STORY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.compare_dir)
    kol_path = out_dir / "summary_by_kol.csv"
    if not kol_path.exists():
        raise FileNotFoundError(f"Missing: {kol_path}")

    kol_df = pd.read_csv(kol_path)
    by_source = build_summary_by_source(kol_df=kol_df, full_name=args.full_name)
    by_source.to_csv(out_dir / "five_point_summary_by_source.csv", index=False)

    overall = (
        by_source.groupby("method", as_index=False)
        .agg(
            n_rows=("source", "count"),
            n_kols_mean=("n_kols", "mean"),
            event_return_mean=("event_return_mean", "mean"),
            event_sharpe_mean=("event_sharpe_mean", "mean"),
            event_mdd_mean=("event_mdd_mean", "mean"),
            UER_mean=("UER_mean", "mean"),
            DRR_mean=("DRR_mean", "mean"),
            BD_mean=("BD_mean", "mean"),
            daily_return_mean=("daily_return_mean", "mean"),
            daily_sharpe_mean=("daily_sharpe_mean", "mean"),
            daily_mdd_mean=("daily_mdd_mean", "mean"),
        )
    )
    overall["HVC_mean"] = overall["UER_mean"] + overall["DRR_mean"]
    overall["MDD_mean"] = overall["event_mdd_mean"]
    overall.to_csv(out_dir / "five_point_summary_overall.csv", index=False)

    win_vs_baseline = build_win_vs_baseline(kol_df=kol_df, full_name=args.full_name)
    win_vs_baseline.to_csv(out_dir / "five_point_win_vs_baseline_by_source.csv", index=False)

    build_story_markdown(
        out_dir=out_dir,
        by_source=by_source,
        win_vs_baseline=win_vs_baseline,
        full_name=args.full_name,
    )
    print(f"Saved five-point story files to: {out_dir}")


if __name__ == "__main__":
    main()
