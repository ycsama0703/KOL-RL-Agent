#!/usr/bin/env python
"""Summarize betrayal_metrics across multiple test directories."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize betrayal metrics from metrics_test.json files.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="List of NAME=DIR pairs, e.g. full=outputs/ablation/test/full",
    )
    parser.add_argument("--output", default="outputs/ablation/summary/betrayal_summary.csv")
    return parser.parse_args()


def latest_run_dir(root: Path, kol: str) -> Path | None:
    pattern = re.compile(rf"^{re.escape(kol)}_\d{{8}}_\d{{6}}$")
    candidates = [p for p in root.iterdir() if p.is_dir() and pattern.match(p.name)]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]


def list_kols(root: Path) -> set[str]:
    pattern = re.compile(r"^(.*)_\d{8}_\d{6}$")
    kols: set[str] = set()
    if not root.exists():
        return kols
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = pattern.match(path.name)
        if match:
            kols.add(match.group(1))
    return kols


def read_metrics(run_dir: Path) -> dict | None:
    path = run_dir / "event" / "metrics_test.json"
    if not path.exists():
        path = run_dir / "metrics_test.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("betrayal_metrics")


def main() -> None:
    args = parse_args()
    pairs = {}
    for item in args.inputs:
        if "=" not in item:
            raise SystemExit(f"Invalid input: {item} (expected NAME=DIR)")
        name, path = item.split("=", 1)
        pairs[name] = Path(path)

    # union of KOLs from all inputs
    kols = set()
    for root in pairs.values():
        kols |= list_kols(root)
    if not kols:
        raise SystemExit("No KOL runs found under the given inputs.")

    rows = []
    for name, root in pairs.items():
        if not root.exists():
            print(f"Skip {name} (missing dir: {root})")
            continue
        for kol in sorted(kols):
            run = latest_run_dir(root, kol)
            if not run:
                continue
            metrics = read_metrics(run)
            if not metrics:
                continue
            row = {"kol": kol, "model": name}
            row.update(metrics)
            rows.append(row)

    if not rows:
        raise SystemExit("No betrayal_metrics found. Did you re-run evaluate_run?")

    df = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["kol", "model"]).to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
