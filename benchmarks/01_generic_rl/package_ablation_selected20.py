#!/usr/bin/env python3
"""Package selected-20 ablation test results into a single folder.

Default behavior:
- 7 variants are read from outputs/ablation/kicl_test_selected20
- w_no_hard and w_no_soft are read from outputs/ablation/kicl_test_allkols
- target KOL list is inferred from selected20 meta csv (method == KICL)
- one latest run directory per (variant, source, kol) is copied

Outputs:
- <output_root>/<variant>/<source>/<run_dir>/...
- <output_root>/manifest.csv
- <output_root>/summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


DEFAULT_VARIANTS: Sequence[str] = (
    "w_no_hard",
    "w_no_soft",
    "w_no_bc_anchor",
    "w_no_rl_completion",
    "w_no_fidelity",
    "w_no_reversal_penalty",
    "w_no_entry_penalty",
    "w_no_market_factors",
    "w_single_head_no_regime_split",
)

RUN_NAME_RE = re.compile(r"^(?P<kol>.+)_20\d{6}_\d{6}$")


@dataclass
class Record:
    variant: str
    source: str
    kol: str
    status: str
    source_run: str
    target_run: str
    message: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--selected20-csv",
        type=Path,
        default=Path("benchmarks/compare/meta/selected20_all_methods_vs_baseline_detailed.csv"),
        help="Selected-20 detail csv; targets are inferred from method==KICL.",
    )
    p.add_argument(
        "--selected20-root",
        type=Path,
        default=Path("outputs/ablation/kicl_test_selected20"),
        help="Ablation test root for selected20 variants.",
    )
    p.add_argument(
        "--allkols-root",
        type=Path,
        default=Path("outputs/ablation/kicl_test_allkols"),
        help="Ablation test root for all-kols variants (used by default for no_hard/no_soft).",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/ablation/kicl_test_pack_selected20"),
        help="Output package root.",
    )
    p.add_argument(
        "--variants",
        type=str,
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated variant list to package.",
    )
    p.add_argument(
        "--copy-mode",
        choices=("copy", "symlink"),
        default="copy",
        help="copy: physical copy; symlink: lightweight links.",
    )
    p.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete output-root before packaging.",
    )
    p.add_argument(
        "--make-tar",
        action="store_true",
        help="Also create <output-root>.tgz",
    )
    return p.parse_args()


def load_targets(path: Path) -> List[Tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"selected20 csv not found: {path}")
    df = pd.read_csv(path)
    for c in ("source", "kol", "method"):
        if c not in df.columns:
            raise ValueError(f"selected20 csv missing column: {c}")
    tgt = (
        df[df["method"] == "KICL"][["source", "kol"]]
        .drop_duplicates()
        .sort_values(["source", "kol"])
    )
    return list(tgt.itertuples(index=False, name=None))


def iter_variant_runs(variant_root: Path, source: str) -> Iterable[Path]:
    # Compatible with both structures:
    # - <variant>/<source>/<run_dir>
    # - <variant>/<source>/<source>/<run_dir>
    p1 = variant_root / source
    p2 = variant_root / source / source
    if p1.exists():
        for d in p1.iterdir():
            if d.is_dir():
                yield d
    if p2.exists():
        for d in p2.iterdir():
            if d.is_dir():
                yield d


def latest_run_for_kol(variant_root: Path, source: str, kol: str) -> Optional[Path]:
    cands: List[Path] = []
    for d in iter_variant_runs(variant_root, source):
        m = RUN_NAME_RE.match(d.name)
        if not m:
            continue
        if m.group("kol") == kol:
            cands.append(d)
    if not cands:
        return None
    return sorted(cands, key=lambda x: x.name, reverse=True)[0]


def is_valid_test_run(run_dir: Path) -> Tuple[bool, str]:
    metric = run_dir / "event" / "metrics_test.json"
    if not metric.exists():
        return False, "missing event/metrics_test.json"
    return True, ""


def copy_run(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copytree(src, dst)
    else:
        dst.symlink_to(src.resolve())


def candidate_variant_roots(
    variant: str, selected20_root: Path, allkols_root: Path
) -> List[Path]:
    # For w_no_hard / w_no_soft, prefer all-kols root but fall back to selected20 root.
    if variant in {"w_no_hard", "w_no_soft"}:
        return [allkols_root / variant, selected20_root / variant]
    return [selected20_root / variant]


def write_manifest(path: Path, rows: Sequence[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "variant",
                "source",
                "kol",
                "status",
                "source_run",
                "target_run",
                "message",
            ]
        )
        for r in rows:
            w.writerow([r.variant, r.source, r.kol, r.status, r.source_run, r.target_run, r.message])


def main() -> None:
    args = parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    targets = load_targets(args.selected20_csv)

    if args.clean_output and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    rows: List[Record] = []

    for variant in variants:
        variant_roots = candidate_variant_roots(
            variant, args.selected20_root, args.allkols_root
        )
        for source, kol in targets:
            run = None
            hit_root = None
            for vroot in variant_roots:
                if not vroot.exists():
                    continue
                run = latest_run_for_kol(vroot, source, kol)
                if run is not None:
                    hit_root = vroot
                    break
            if run is None:
                rows.append(
                    Record(
                        variant=variant,
                        source=source,
                        kol=kol,
                        status="missing_run",
                        source_run="",
                        target_run="",
                        message=f"not found under any of: {', '.join(str(x) for x in variant_roots)}",
                    )
                )
                continue

            ok, msg = is_valid_test_run(run)
            if not ok:
                rows.append(
                    Record(
                        variant=variant,
                        source=source,
                        kol=kol,
                        status="invalid_run",
                        source_run=str(run),
                        target_run="",
                        message=f"{msg}; root={hit_root}",
                    )
                )
                continue

            target_run = args.output_root / variant / source / run.name
            if target_run.exists():
                if target_run.is_symlink() or target_run.is_file():
                    target_run.unlink()
                else:
                    shutil.rmtree(target_run)
            copy_run(run, target_run, args.copy_mode)
            rows.append(
                Record(
                    variant=variant,
                    source=source,
                    kol=kol,
                    status="copied",
                    source_run=str(run),
                    target_run=str(target_run),
                    message="",
                )
            )

    manifest_path = args.output_root / "manifest.csv"
    write_manifest(manifest_path, rows)

    copied = [r for r in rows if r.status == "copied"]
    missing = [r for r in rows if r.status == "missing_run"]
    invalid = [r for r in rows if r.status == "invalid_run"]
    summary = {
        "targets": len(targets),
        "variants": variants,
        "expected_total": len(targets) * len(variants),
        "copied_total": len(copied),
        "missing_run_total": len(missing),
        "invalid_run_total": len(invalid),
        "output_root": str(args.output_root),
        "manifest": str(manifest_path),
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if missing:
        print("\nMissing examples:")
        for r in missing[:8]:
            print(f"  {r.variant} {r.source}/{r.kol} -> {r.message}")
    if invalid:
        print("\nInvalid examples:")
        for r in invalid[:8]:
            print(f"  {r.variant} {r.source}/{r.kol} -> {r.message}")

    if args.make_tar:
        tar_path = args.output_root.with_suffix(".tgz")
        if tar_path.exists():
            tar_path.unlink()
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(args.output_root, arcname=args.output_root.name)
        print(f"\nTar created: {tar_path}")


if __name__ == "__main__":
    main()
