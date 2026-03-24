"""Build FULL-vs-ablation comparison report (independent entrypoint).

This script is intentionally separate from run_ablation_kicl.sh so that
ablation compare can be generated directly from already-tested result folders.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from build_compare_report import (
    build_key_set,
    discover_latest_runs,
    method_styles,
    summarize_by_method,
    summarize_by_method_by_source,
    write_overview_plots,
    write_per_kol_outputs,
)


DEFAULT_VARIANT_LABELS = {
    "w_no_hard": "WO_HARD",
    "w_no_soft": "WO_SOFT",
    "w_no_bc_anchor": "WO_BC_ANCHOR",
    "w_no_rl_completion": "WO_RL_COMPLETION",
    "w_no_fidelity": "WO_FIDELITY",
    "w_no_reversal_penalty": "WO_REV_PEN",
    "w_no_entry_penalty": "WO_ENTRY_PEN",
    "w_no_market_factors": "WO_MKT_FACTORS",
    "w_single_head_no_regime_split": "WO_REGIME_SPLIT",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare FULL model with ablation variants (benchmark-style report)."
    )
    p.add_argument(
        "--full-root",
        required=True,
        help="FULL/KICL test root (contains x/ and youtube/ run folders).",
    )
    p.add_argument(
        "--full-name",
        default="FULL",
        help="Display name for full model.",
    )
    p.add_argument(
        "--ablation-root",
        required=True,
        help="Ablation test root containing variant folders.",
    )
    p.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Variant folder name under --ablation-root. Repeatable. Default: auto-discover all.",
    )
    p.add_argument(
        "--output-dir",
        default="benchmarks/compare/ablation_vs_full",
        help="Output directory for compare report.",
    )
    p.add_argument(
        "--mode",
        choices=["anchor_ours", "intersection", "union"],
        default="intersection",
        help=(
            "KOL key selection: intersection is recommended for ablation comparisons "
            "to ensure all methods are directly comparable."
        ),
    )
    p.add_argument(
        "--source-filter",
        choices=["all", "x", "youtube", "x,youtube", "youtube,x"],
        default="all",
        help="Restrict comparison to one/both sources.",
    )
    p.add_argument(
        "--plot-format",
        choices=["png", "pdf"],
        default="png",
    )
    p.add_argument(
        "--event-curve-mode",
        choices=["daily_mtm", "signal_step"],
        default="daily_mtm",
    )
    p.add_argument(
        "--include-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument(
        "--highlight-method",
        default=None,
        help="Method name to highlight. Default: --full-name",
    )
    p.add_argument("--highlight-color", default="#F39C12")
    p.add_argument("--highlight-linewidth", type=float, default=3.2)
    p.add_argument("--other-linewidth", type=float, default=1.8)
    p.add_argument("--other-alpha", type=float, default=1.0)
    return p.parse_args()


def discover_variants(ablation_root: Path, specified: List[str]) -> List[str]:
    if specified:
        variants = specified
    else:
        variants = sorted([p.name for p in ablation_root.iterdir() if p.is_dir()])
    if not variants:
        raise ValueError(f"No variants found under {ablation_root}")
    missing = [v for v in variants if not (ablation_root / v).is_dir()]
    if missing:
        raise FileNotFoundError(f"Variant dirs not found: {missing}")
    return variants


def label_for_variant(variant: str) -> str:
    return DEFAULT_VARIANT_LABELS.get(variant, variant.upper())


def filter_keys_by_source(
    keys: List[Tuple[str, str]],
    source_filter: str,
) -> List[Tuple[str, str]]:
    if source_filter == "all":
        return keys
    allowed = {s.strip() for s in source_filter.split(",")}
    return [k for k in keys if k[0] in allowed]


def main() -> None:
    args = parse_args()

    full_root = Path(args.full_root)
    ablation_root = Path(args.ablation_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not full_root.exists():
        raise FileNotFoundError(f"FULL root not found: {full_root}")
    if not ablation_root.exists():
        raise FileNotFoundError(f"Ablation root not found: {ablation_root}")

    variants = discover_variants(ablation_root, args.variant)

    methods: Dict[str, Path] = {args.full_name: full_root}
    variant_label_map: Dict[str, str] = {}
    for variant in variants:
        label = label_for_variant(variant)
        methods[label] = ablation_root / variant
        variant_label_map[variant] = label

    method_order = list(methods.keys())
    anchor_method = args.full_name
    highlight_method = args.highlight_method or args.full_name
    styles = method_styles(
        method_order=method_order,
        highlight_method=highlight_method,
        highlight_color=args.highlight_color,
        highlight_linewidth=args.highlight_linewidth,
        other_linewidth=args.other_linewidth,
        other_alpha=args.other_alpha,
    )

    method_runs = {name: discover_latest_runs(root) for name, root in methods.items()}
    keys = build_key_set(args.mode, method_runs, anchor_method=anchor_method)
    keys = filter_keys_by_source(keys, args.source_filter)

    summary_rows = []
    for key in keys:
        summary_rows.append(
            write_per_kol_outputs(
                key=key,
                method_order=method_order,
                method_runs=method_runs,
                output_root=out_dir,
                plot_format=args.plot_format,
                event_curve_mode=args.event_curve_mode,
                styles=styles,
                include_baseline=args.include_baseline,
            )
        )

    if not summary_rows:
        raise RuntimeError("No KOL entries selected. Check roots/mode/source-filter.")

    import pandas as pd  # local import to keep startup small

    summary_df = pd.DataFrame(summary_rows).sort_values(["source", "kol"])
    summary_df.to_csv(out_dir / "summary_by_kol.csv", index=False)

    method_summary = summarize_by_method(summary_df, method_order)
    method_summary.to_csv(out_dir / "summary_by_method_mean.csv", index=False)
    by_source = summarize_by_method_by_source(summary_df, method_order)
    by_source.to_csv(out_dir / "summary_by_method_mean_by_source.csv", index=False)
    write_overview_plots(summary_df, method_order, out_dir)

    meta = {
        "full_name": args.full_name,
        "full_root": str(full_root),
        "ablation_root": str(ablation_root),
        "variants": variants,
        "variant_label_map": variant_label_map,
        "mode": args.mode,
        "source_filter": args.source_filter,
        "event_curve_mode": args.event_curve_mode,
        "include_baseline": args.include_baseline,
        "highlight_method": highlight_method,
        "methods": {k: str(v) for k, v in methods.items()},
        "n_kols_total": int(len(summary_df)),
        "output_dir": str(out_dir),
    }
    with (out_dir / "compare_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved ablation comparison to: {out_dir}")
    print(f"Total KOL entries: {len(summary_df)}")
    print("Methods:", ", ".join(method_order))


if __name__ == "__main__":
    main()

