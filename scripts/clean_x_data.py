#!/usr/bin/env python3
"""Clean X/Twitter jsonl into a compact schema for LLM labeling and analysis.

Key transformations:
- Drop heavy profile/rendering fields while keeping core text, media, and engagement data
- Normalize created_at to UTC ISO8601 (Z)
- Optionally merge quoted/retweeted text into a single `text` field
- Extract tickers from entities.symbols and cashtags in text

Input can be:
- A directory containing per-KOL *.jsonl files
- A single *.jsonl file

Output:
- Writes cleaned *.jsonl files into output_dir (flat layout) OR
  `output_dir/<kol_username>/<tweet_type>.jsonl` (kol/type layout)
- Writes a manifest JSON with per-file counts + date ranges
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, TextIO, Tuple


TWITTER_TIME_FMT = "%a %b %d %H:%M:%S %z %Y"
CASHTAG_RE = re.compile(r"\\$([A-Za-z][A-Za-z0-9_.\\-]{0,14})")

LayoutMode = Literal["flat", "kol_type"]
TextMode = Literal["author", "combined", "labeled_combined"]


@dataclass(frozen=True)
class CleanConfig:
    input_path: Path
    output_dir: Path
    include_quote_text: bool
    include_retweet_text: bool
    layout: LayoutMode
    text_mode: TextMode


def parse_args() -> CleanConfig:
    p = argparse.ArgumentParser(description="Clean X/Twitter jsonl into minimal schema.")
    p.add_argument("--input", default="data/x_data/processed", help="Input dir or jsonl file.")
    p.add_argument("--output-dir", default="data/x_data/cleaned", help="Output directory.")
    p.add_argument(
        "--layout",
        choices=("flat", "kol_type"),
        default="flat",
        help="Output layout: flat=<output-dir>/*.jsonl, kol_type=<output-dir>/<kol>/<tweet_type>.jsonl",
    )
    p.add_argument(
        "--text-mode",
        choices=("author", "combined", "labeled_combined"),
        default="author",
        help=(
            "How to populate the top-level `text` field: "
            "author=KOL-authored text only, combined=main+quoted+retweeted, labeled_combined=combined with labels."
        ),
    )
    p.add_argument("--no-quote-text", action="store_true", help="Do not include quoted_tweet text.")
    p.add_argument("--no-retweet-text", action="store_true", help="Do not include retweeted_tweet text.")
    args = p.parse_args()
    return CleanConfig(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        include_quote_text=not bool(args.no_quote_text),
        include_retweet_text=not bool(args.no_retweet_text),
        layout=args.layout,
        text_mode=args.text_mode,
    )


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                continue


def get_kol(obj: Dict[str, Any]) -> str:
    return str(obj.get("kol_username") or obj.get("tweet", {}).get("author", {}).get("userName") or "__MISSING_KOL__")


def parse_created_at(obj: Dict[str, Any]) -> datetime | None:
    created = obj.get("created_at") or obj.get("tweet", {}).get("createdAt")
    if not created:
        return None
    try:
        return datetime.strptime(created, TWITTER_TIME_FMT).astimezone(timezone.utc)
    except Exception:
        return None


def iso_z(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def extract_symbols(tweet_obj: Dict[str, Any]) -> List[str]:
    entities = tweet_obj.get("entities") or {}
    symbols = entities.get("symbols") or []
    out: List[str] = []
    for item in symbols:
        text = item.get("text")
        if text:
            out.append(str(text))
    return out


def extract_cashtags(text: str) -> List[str]:
    return [m.group(1) for m in CASHTAG_RE.finditer(text)]


def sanitize_ticker(ticker: str) -> str:
    return ticker.strip().lstrip("$").upper().replace(".", "-")


def safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def build_text(tweet: Dict[str, Any], *, include_quote_text: bool, include_retweet_text: bool) -> Tuple[str, str | None, str | None]:
    main = tweet.get("text") if isinstance(tweet.get("text"), str) else ""
    main = main.strip()

    quoted = None
    if include_quote_text:
        qt = tweet.get("quoted_tweet") or {}
        qtxt = qt.get("text")
        if isinstance(qtxt, str) and qtxt.strip():
            quoted = qtxt.strip()

    retweeted = None
    if include_retweet_text:
        rt = tweet.get("retweeted_tweet") or {}
        rtxt = rt.get("text")
        if isinstance(rtxt, str) and rtxt.strip():
            retweeted = rtxt.strip()

    parts = [p for p in [main, quoted, retweeted] if p]
    combined = "\n\n".join(parts)
    return combined, quoted, retweeted


def extract_media_items(obj: Dict[str, Any], tweet: Dict[str, Any]) -> List[Dict[str, str]]:
    raw_items: List[Dict[str, Any]] = []
    top_media = obj.get("media")
    if isinstance(top_media, list):
        raw_items.extend(item for item in top_media if isinstance(item, dict))

    ext_media = ((tweet.get("extendedEntities") or {}).get("media")) or []
    if isinstance(ext_media, list):
        raw_items.extend(item for item in ext_media if isinstance(item, dict))

    dedup: Dict[tuple[str, str], Dict[str, str]] = {}
    for item in raw_items:
        kind = str(item.get("kind") or item.get("type") or "unknown")
        url = item.get("url") or item.get("media_url_https") or item.get("media_url")
        if not url:
            continue
        key = (kind, str(url))
        dedup[key] = {"kind": kind, "url": str(url)}
    return list(dedup.values())


def extract_tickers_main(tweet: Dict[str, Any], main_text: str) -> List[str]:
    tickers: List[str] = []
    tickers.extend(extract_symbols(tweet))
    tickers.extend(extract_cashtags(main_text))
    norm = [sanitize_ticker(t) for t in tickers if t]
    return sorted(set(norm))


def extract_tickers_context(tweet: Dict[str, Any], context_text: str) -> List[str]:
    tickers: List[str] = []
    for nested_key in ("quoted_tweet", "retweeted_tweet"):
        nested = tweet.get(nested_key) or {}
        tickers.extend(extract_symbols(nested))
    tickers.extend(extract_cashtags(context_text))
    norm = [sanitize_ticker(t) for t in tickers if t]
    return sorted(set(norm))


def output_name_for_input(path: Path) -> str:
    # Keep per-KOL naming when input already per-KOL; otherwise single file.
    if path.suffix == ".jsonl":
        return path.name
    return "cleaned.jsonl"


def safe_segment(name: str, *, fallback: str) -> str:
    raw = (name or "").strip()
    if not raw:
        return fallback
    # Keep it filesystem-safe and stable
    return re.sub(r"[^A-Za-z0-9_\\-]+", "_", raw)


@dataclass
class FileStats:
    rows: int = 0
    min_dt: datetime | None = None
    max_dt: datetime | None = None


def update_stats(stats: FileStats, dt: datetime | None) -> None:
    stats.rows += 1
    if dt is None:
        return
    if stats.min_dt is None or dt < stats.min_dt:
        stats.min_dt = dt
    if stats.max_dt is None or dt > stats.max_dt:
        stats.max_dt = dt


def pick_text_field(text_mode: TextMode, *, main: str, quoted: str | None, retweeted: str | None) -> str:
    if text_mode == "author":
        return main
    if text_mode == "combined":
        parts = [p for p in [main, quoted, retweeted] if p]
        return "\n\n".join(parts)
    # labeled_combined
    parts: List[str] = []
    if main:
        parts.append("AUTHOR_TEXT:\n" + main)
    if quoted:
        parts.append("QUOTED_TEXT (context, not author stance):\n" + quoted)
    if retweeted:
        parts.append("RETWEETED_TEXT (context, not author stance):\n" + retweeted)
    return "\n\n".join(parts)


def main() -> None:
    cfg = parse_args()
    if not cfg.input_path.exists():
        raise SystemExit(f"Input not found: {cfg.input_path}")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    inputs: List[Path] = []
    if cfg.input_path.is_dir():
        inputs = sorted([p for p in cfg.input_path.iterdir() if p.is_file() and p.suffix == ".jsonl"])
    else:
        inputs = [cfg.input_path]

    if not inputs:
        raise SystemExit(f"No .jsonl files found under {cfg.input_path}")

    manifest: Dict[str, Any] = {
        "input": str(cfg.input_path),
        "output_dir": str(cfg.output_dir),
        "include_quote_text": cfg.include_quote_text,
        "include_retweet_text": cfg.include_retweet_text,
        "layout": cfg.layout,
        "text_mode": cfg.text_mode,
        "files": [],
    }

    for in_path in inputs:
        # Lazily-open writers per output file to support kol/type layout without
        # reopening for every line.
        writers: Dict[Path, TextIO] = {}
        stats_by_out: Dict[Path, FileStats] = {}

        def get_writer(obj: Dict[str, Any]) -> Path:
            if cfg.layout == "flat":
                out_path = cfg.output_dir / output_name_for_input(in_path)
            else:
                kol = safe_segment(get_kol(obj), fallback="__MISSING_KOL__")
                ttype_raw = obj.get("tweet_type")
                ttype = safe_segment(str(ttype_raw) if ttype_raw is not None else "", fallback="unknown")
                out_path = cfg.output_dir / kol / f"{ttype}.jsonl"
                out_path.parent.mkdir(parents=True, exist_ok=True)

            if out_path not in writers:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                writers[out_path] = out_path.open("w", encoding="utf-8")
                stats_by_out[out_path] = FileStats()
            return out_path

        try:
            for obj in iter_jsonl(in_path):
                out_path = get_writer(obj)
                out = writers[out_path]
                tweet = obj.get("tweet") or {}
                created_utc = parse_created_at(obj)

                combined, quoted, retweeted = build_text(
                    tweet,
                    include_quote_text=cfg.include_quote_text,
                    include_retweet_text=cfg.include_retweet_text,
                )

                main_text = (tweet.get("text") or "").strip() if isinstance(tweet.get("text"), str) else ""
                context_parts = [p for p in [quoted, retweeted] if p]
                context_text = "\n\n".join(context_parts)

                tickers_main = extract_tickers_main(tweet, main_text)
                tickers_context = extract_tickers_context(tweet, context_text)
                tickers_all = sorted(set(tickers_main + tickers_context))

                cleaned = {
                    "platform": "x",
                    "kol_username": get_kol(obj),
                    "tweet_id": str(tweet.get("id") or ""),
                    "tweet_type": obj.get("tweet_type"),
                    "created_at_utc": iso_z(created_utc),
                    "fetched_at_utc": obj.get("fetched_at_utc"),
                    "lang": tweet.get("lang"),
                    "url": tweet.get("url") or tweet.get("twitterUrl"),
                    "in_reply_to_username": tweet.get("inReplyToUsername"),
                    # Keep compatibility fields but make author/context explicit.
                    "text": pick_text_field(cfg.text_mode, main=main_text, quoted=quoted, retweeted=retweeted),
                    "text_main": main_text,
                    "text_quoted": quoted,
                    "text_retweeted": retweeted,
                    "tickers": tickers_all,
                    "tickers_main": tickers_main,
                    "tickers_context": tickers_context,
                    "engagement": {
                        "like_count": safe_int(tweet.get("likeCount")),
                        "reply_count": safe_int(tweet.get("replyCount")),
                        "retweet_count": safe_int(tweet.get("retweetCount")),
                        "quote_count": safe_int(tweet.get("quoteCount")),
                        "view_count": safe_int(tweet.get("viewCount")),
                        "bookmark_count": safe_int(tweet.get("bookmarkCount")),
                    },
                    "author": {
                        "username": tweet.get("author", {}).get("userName"),
                        "name": tweet.get("author", {}).get("name"),
                        "followers": safe_int(tweet.get("author", {}).get("followers")),
                    },
                    "media": extract_media_items(obj, tweet),
                }
                out.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
                update_stats(stats_by_out[out_path], created_utc)
        finally:
            for fp in writers.values():
                try:
                    fp.close()
                except Exception:
                    pass

        for out_path, st in sorted(stats_by_out.items(), key=lambda x: str(x[0])):
            manifest["files"].append(
                {
                    "input": str(in_path),
                    "output": str(out_path),
                    "rows": st.rows,
                    "date_min_utc": iso_z(st.min_dt),
                    "date_max_utc": iso_z(st.max_dt),
                }
            )

    (cfg.output_dir / "manifest_cleaned.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Cleaned {len(inputs)} file(s) -> {cfg.output_dir}")


if __name__ == "__main__":
    main()
