"""Step-03: split unified schema datasets into train/val/test by trading day.

Input (default):
  data/multisource_ready_22-25/02_unified_schema/{youtube,x}/*.csv

Output:
  data/multisource_ready_22-25/03_splits/{youtube,x}/{KOL}/{train,val,test}.csv
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


@dataclass(frozen=True)
class Config:
    input_root: Path
    output_root: Path
    train_ratio: float
    val_ratio: float
    test_ratio: float


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Prepare step-03 train/val/test splits by trading day.")
    p.add_argument("--input-root", default="data/multisource_ready_22-25/02_unified_schema")
    p.add_argument("--output-root", default="data/multisource_ready_22-25/03_splits")
    p.add_argument("--train-ratio", type=float, default=0.6)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--test-ratio", type=float, default=0.2)
    args = p.parse_args()
    return Config(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
    )


def safe_name(text: str) -> str:
    out = re.sub(r"[^\w\-\.]+", "_", text.strip())
    return out or "UNKNOWN_KOL"


def compute_day_counts(total: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("Ratios must sum to 1.0")

    tr = int(total * train_ratio)
    va = int(total * val_ratio)
    te = total - tr - va

    # Keep temporal split valid while trying to keep each split non-empty when possible.
    if total >= 3:
        if tr <= 0:
            tr = 1
        if va <= 0:
            va = 1
        te = total - tr - va
        if te <= 0:
            if tr > va and tr > 1:
                tr -= 1
            elif va > 1:
                va -= 1
            te = total - tr - va
            if te <= 0:
                te = 1
                if tr > 1:
                    tr -= 1
                elif va > 1:
                    va -= 1
    elif total == 2:
        tr, va, te = 1, 0, 1
    else:  # total == 1
        tr, va, te = 1, 0, 0
    return tr, va, te


def split_one_file(path: Path, out_source_dir: Path, cfg: Config) -> Dict[str, object]:
    df = pd.read_csv(path)
    if "trading_day" not in df.columns:
        return {"file": path.name, "skipped": True, "reason": "missing_trading_day"}

    if "channel_name" in df.columns and not df["channel_name"].dropna().empty:
        kol = str(df["channel_name"].dropna().iloc[0]).strip()
    else:
        kol = path.stem.replace("_companies_cleaned", "")
    kol_dir = out_source_dir / safe_name(kol)
    kol_dir.mkdir(parents=True, exist_ok=True)

    days = pd.to_datetime(df["trading_day"], errors="coerce").dropna().dt.strftime("%Y-%m-%d")
    valid = days.notna()
    df = df.loc[valid].copy()
    days = days.loc[valid]
    if df.empty:
        return {"file": path.name, "kol": kol, "skipped": True, "reason": "no_valid_days"}

    df["_trading_day_norm"] = days.values
    if "published_at" in df.columns:
        df["_published_at_norm"] = pd.to_datetime(df["published_at"], errors="coerce")
    else:
        df["_published_at_norm"] = pd.NaT
    df = df.sort_values(["_trading_day_norm", "_published_at_norm"], kind="stable").reset_index(drop=True)

    unique_days = df["_trading_day_norm"].dropna().drop_duplicates().tolist()
    tr_days, va_days, te_days = compute_day_counts(len(unique_days), cfg.train_ratio, cfg.val_ratio, cfg.test_ratio)

    train_set = set(unique_days[:tr_days])
    val_set = set(unique_days[tr_days : tr_days + va_days])
    test_set = set(unique_days[tr_days + va_days :])

    split_col = []
    for d in df["_trading_day_norm"]:
        if d in train_set:
            split_col.append("train")
        elif d in val_set:
            split_col.append("val")
        else:
            split_col.append("test")
    df["_split"] = split_col

    stats = {
        "file": path.name,
        "kol": kol,
        "rows": int(len(df)),
        "unique_days": int(len(unique_days)),
        "train_days": int(tr_days),
        "val_days": int(va_days),
        "test_days": int(te_days),
        "splits": {},
    }

    for split in ["train", "val", "test"]:
        sdf = df[df["_split"] == split].drop(columns=["_trading_day_norm", "_published_at_norm", "_split"])
        out_path = kol_dir / f"{split}.csv"
        sdf.to_csv(out_path, index=False)
        stats["splits"][split] = {
            "rows": int(len(sdf)),
            "day_min": str(sdf["trading_day"].min()) if len(sdf) else None,
            "day_max": str(sdf["trading_day"].max()) if len(sdf) else None,
            "output": str(out_path),
        }
    print(
        f"{path.name} -> {kol_dir} | days(train/val/test)="
        f"{tr_days}/{va_days}/{te_days} rows="
        f"{stats['splits']['train']['rows']}/{stats['splits']['val']['rows']}/{stats['splits']['test']['rows']}"
    )
    return stats


def process_source(source: str, in_dir: Path, out_dir: Path, cfg: Config) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in in_dir.glob("*.csv") if p.is_file()])
    details: List[Dict[str, object]] = []
    for path in files:
        details.append(split_one_file(path, out_dir, cfg))
    return {
        "source": source,
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "file_count": len(files),
        "files": details,
    }


def main() -> None:
    cfg = parse_args()
    if not cfg.input_root.exists():
        raise SystemExit(f"Input root not found: {cfg.input_root}")

    manifest = {
        "task": "prepare_step03_daily_split",
        "config": {
            **asdict(cfg),
            "input_root": str(cfg.input_root),
            "output_root": str(cfg.output_root),
            "split_rule": "chronological by trading_day, same-day samples stay in same split",
        },
        "sources": [],
    }

    for source in ["youtube", "x"]:
        in_dir = cfg.input_root / source
        if not in_dir.exists():
            print(f"Skip source={source}: missing {in_dir}")
            continue
        out_dir = cfg.output_root / source
        manifest["sources"].append(process_source(source, in_dir, out_dir, cfg))

    cfg.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = cfg.output_root / "manifest_03_splits.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
