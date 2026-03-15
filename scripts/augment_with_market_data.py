"""Augment cleaned datasets with embeddings and compact market factors.

Exposed market factors (no future leakage; computed up to sample timestamp):
- ret_1d
- ret_5d
- vol_5d
- vol_20d
- volu_z_20d
- dist_sma20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from pandas.tseries.offsets import BDay

# add repo root for local imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.market.company_mapper import CompanyTickerMapper


LOGGER = logging.getLogger("augment")
MARKET_FACTOR_COLS = ["ret_1d", "ret_5d", "vol_5d", "vol_20d", "volu_z_20d", "dist_sma20"]


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create enriched datasets with embeddings + market factors.")
    parser.add_argument(
        "--input",
        default="data/processed/cleaned",
        help="Root directory containing CSV files (supports recursive subdirectories).",
    )
    parser.add_argument(
        "--embeddings",
        default="data/processed/embeddings",
        help="Root directory containing PT files with the same relative paths as input CSVs.",
    )
    parser.add_argument("--output", default="data/processed/enriched", help="Destination root directory for augmented CSVs.")
    parser.add_argument(
        "--ticker-reference",
        default="data/input/top_500_companies_list.xlsx",
        help="Excel file containing at least [Symbol, Company Name] columns.",
    )
    parser.add_argument(
        "--alias-overrides",
        default="config/company_alias_overrides.csv",
        help="CSV file with alias overrides: columns [alias,ticker]. Empty ticker means drop (None).",
    )
    parser.add_argument(
        "--historical-lexicon-root",
        default="data",
        help="Root directory to mine historical company->ticker pairs from CSV files.",
    )
    parser.add_argument(
        "--historical-lexicon-min-support",
        type=int,
        default=1,
        help="Minimum count for a historical company->ticker pair to be used.",
    )
    parser.add_argument(
        "--disable-historical-lexicon",
        action="store_true",
        help="Disable historical company->ticker fallback.",
    )
    parser.add_argument(
        "--ticker-vocab-paths",
        default="models/embedding/ticker_vocab.json,models/embedding/22-24_ticker_vocab.json",
        help="Comma-separated JSON vocab paths used by short-ticker fallback.",
    )
    parser.add_argument(
        "--disable-short-ticker-fallback",
        action="store_true",
        help="Disable fallback that treats company as ticker when it is a valid short symbol.",
    )
    parser.add_argument(
        "--channels",
        default=None,
        help="Optional comma-separated allowlist for first-level folders under --input (default all).",
    )
    parser.add_argument(
        "--price-days",
        type=int,
        default=5,
        help="Legacy arg kept for compatibility; factor windows are fixed at 5/20.",
    )
    parser.add_argument("--chunk-size", type=int, default=20, help="Number of tickers per Yahoo Finance download chunk.")
    return parser.parse_args()


def chunk_list(items: Sequence[str], size: int) -> Iterable[List[str]]:
    for idx in range(0, len(items), size):
        yield list(items[idx : idx + size])


def sanitize_ticker(ticker: str) -> str:
    return ticker.replace(".", "-")


def load_alias_overrides(path: Path) -> Dict[str, str | None]:
    if not path.exists():
        LOGGER.info("Alias overrides not found: %s (skip)", path)
        return {}
    df = pd.read_csv(path)
    if "alias" not in df.columns or "ticker" not in df.columns:
        raise ValueError(f"Alias file {path} must contain columns: alias,ticker")
    out: Dict[str, str | None] = {}
    for _, row in df.iterrows():
        alias = str(row["alias"]).strip()
        if not alias:
            continue
        ticker_raw = row["ticker"]
        if pd.isna(ticker_raw):
            out[alias] = None
            continue
        ticker = str(ticker_raw).strip().upper()
        out[alias] = ticker if ticker else None
    LOGGER.info("Loaded %d alias overrides from %s", len(out), path)
    return out


def normalize_company(text: str) -> str:
    return CompanyTickerMapper._normalize(text)  # noqa: SLF001


def load_ticker_vocab_symbols(paths: Sequence[Path]) -> Set[str]:
    symbols: Set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to parse ticker vocab %s: %s", path, exc)
            continue
        keys = payload.keys() if isinstance(payload, dict) else payload
        for key in keys:
            symbol = str(key).strip().upper()
            if not symbol or symbol in {"<PAD>", "<UNK>"}:
                continue
            symbols.add(symbol)
    return symbols


def build_historical_lexicon(
    root: Path,
    min_support: int,
    excluded_prefixes: Sequence[Path],
) -> Dict[str, str]:
    if not root.exists():
        LOGGER.info("Historical lexicon root not found: %s (skip)", root)
        return {}

    counters: Dict[str, Counter] = defaultdict(Counter)
    csv_files = sorted(path for path in root.rglob("*.csv") if path.is_file())
    for csv_path in csv_files:
        if any(str(csv_path).startswith(str(prefix)) for prefix in excluded_prefixes):
            continue
        try:
            df = pd.read_csv(csv_path, usecols=lambda c: c in {"company", "ticker"})
        except Exception:
            continue
        if not {"company", "ticker"}.issubset(df.columns):
            continue
        sub = df[["company", "ticker"]].dropna()
        if sub.empty:
            continue
        for _, row in sub.iterrows():
            company_norm = normalize_company(str(row["company"]))
            ticker = str(row["ticker"]).strip().upper()
            if not company_norm:
                continue
            if not ticker or ticker in {"NAN", "NONE"}:
                continue
            counters[company_norm][ticker] += 1

    lexicon: Dict[str, str] = {}
    for company_norm, counter in counters.items():
        ticker, count = counter.most_common(1)[0]
        if count >= min_support:
            lexicon[company_norm] = ticker
    LOGGER.info("Built historical lexicon with %d entries (min_support=%d)", len(lexicon), min_support)
    return lexicon


def _extract_market_frame(data: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    if data.empty:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        close_col = (ticker, "Close")
        vol_col = (ticker, "Volume")
        if close_col not in data.columns:
            return None
        out = pd.DataFrame({"close": data[close_col]})
        if vol_col in data.columns:
            out["volume"] = data[vol_col]
    else:
        if "Close" not in data.columns:
            return None
        out = pd.DataFrame({"close": data["Close"]})
        if "Volume" in data.columns:
            out["volume"] = data["Volume"]
    out = out.dropna(subset=["close"])
    out.index = out.index.tz_localize(None)
    return out


def download_single_ticker(ticker: str, start, end) -> pd.DataFrame | None:
    try:
        data = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=False,
        )
    except Exception as exc:  # pragma: no cover
        LOGGER.error("Failed download for %s: %s", ticker, exc)
        return None
    return _extract_market_frame(data, ticker)


def download_market_panel(tickers: List[str], start, end, chunk_size: int) -> Dict[str, pd.DataFrame]:
    market: Dict[str, pd.DataFrame] = {}
    for chunk in chunk_list(tickers, chunk_size):
        try:
            data = yf.download(
                " ".join(chunk),
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                group_by="ticker",
                threads=True,
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to download chunk %s: %s", chunk, exc)
            data = pd.DataFrame()

        if data.empty:
            for ticker in chunk:
                frame = download_single_ticker(ticker, start, end)
                if frame is not None:
                    market[ticker] = frame
            continue

        if isinstance(data.columns, pd.MultiIndex):
            for ticker in chunk:
                frame = _extract_market_frame(data, ticker)
                if frame is not None and not frame.empty:
                    market[ticker] = frame
                else:
                    fallback = download_single_ticker(ticker, start, end)
                    if fallback is not None:
                        market[ticker] = fallback
        else:
            if len(chunk) == 1:
                frame = _extract_market_frame(data, chunk[0])
                if frame is not None and not frame.empty:
                    market[chunk[0]] = frame
            else:
                # fallback per ticker
                for ticker in chunk:
                    frame = download_single_ticker(ticker, start, end)
                    if frame is not None:
                        market[ticker] = frame
    return market


def compute_market_factors(hist: pd.DataFrame) -> Dict[str, float] | None:
    close = hist["close"].dropna()
    if len(close) < 21:
        return None
    rets = close.pct_change().dropna()
    if len(rets) < 20:
        return None

    ret_1d = float(close.iloc[-1] / close.iloc[-2] - 1.0)
    ret_5d = float(close.iloc[-1] / close.iloc[-6] - 1.0)
    vol_5d = float(rets.tail(5).std(ddof=0))
    vol_20d = float(rets.tail(20).std(ddof=0))

    sma20 = float(close.tail(20).mean())
    dist_sma20 = float(close.iloc[-1] / sma20 - 1.0) if abs(sma20) > 1e-12 else 0.0

    volu_z_20d = 0.0
    if "volume" in hist.columns:
        vol = hist["volume"].dropna()
        if len(vol) >= 20:
            vol_tail = vol.tail(20).astype(float)
            vol_std = float(vol_tail.std(ddof=0))
            if vol_std > 1e-12:
                volu_z_20d = float((vol_tail.iloc[-1] - float(vol_tail.mean())) / vol_std)

    return {
        "ret_1d": ret_1d,
        "ret_5d": ret_5d,
        "vol_5d": vol_5d,
        "vol_20d": vol_20d,
        "volu_z_20d": volu_z_20d,
        "dist_sma20": dist_sma20,
    }


def append_market_factors(
    df: pd.DataFrame,
    market_data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    filled = []
    for _, row in df.iterrows():
        ticker = row["yf_ticker"]
        series = market_data.get(ticker)
        if series is None or series.empty or "close" not in series.columns:
            continue
        publish_ts = row["published_at"]
        if publish_ts.tzinfo is not None:
            publish_ts = publish_ts.tz_convert(None)
        cutoff = pd.Timestamp(publish_ts.date())
        hist = series.loc[:cutoff].copy()
        factors = compute_market_factors(hist)
        if factors is None:
            continue
        enriched = row.to_dict()
        for key, value in factors.items():
            enriched[key] = value
        enriched.pop("yf_ticker", None)
        filled.append(enriched)
    if not filled:
        return pd.DataFrame(columns=df.columns)
    return pd.DataFrame(filled)


def process_file(
    csv_path: Path,
    emb_path: Path,
    output_path: Path,
    mapper: CompanyTickerMapper,
    historical_lexicon: Dict[str, str],
    valid_short_tickers: Set[str],
    enable_short_ticker_fallback: bool,
    price_days: int,
    chunk_size: int,
) -> None:
    df = pd.read_csv(csv_path)
    if df.empty:
        LOGGER.info("Skipping %s (empty)", csv_path)
        return
    if "published_at" not in df.columns and "publishedAt" in df.columns:
        df["published_at"] = df["publishedAt"]
    if "published_at" not in df.columns:
        LOGGER.warning("Missing published_at/publishedAt in %s; skipping.", csv_path)
        return
    if not emb_path.exists():
        LOGGER.warning("Missing embedding file %s; skipping %s", emb_path, csv_path)
        return
    payload = torch.load(emb_path, map_location="cpu")
    embeddings = payload["embeddings"]
    if len(embeddings) != len(df):
        LOGGER.warning("Row mismatch between %s and %s; skipping.", csv_path, emb_path)
        return

    emb_array = embeddings.numpy() if hasattr(embeddings, "numpy") else np.array(embeddings)
    emb_cols = [f"embedding_{idx}" for idx in range(emb_array.shape[1])]
    emb_df = pd.DataFrame(emb_array, columns=emb_cols)

    df = df.reset_index(drop=True)
    df = pd.concat([df, emb_df], axis=1)

    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    company_norm = None
    if "company" in df.columns:
        company_norm = df["company"].astype(str).map(normalize_company)

    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
        df["ticker"] = df["ticker"].replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA})
        # If ticker column exists but is missing for many rows (common in YouTube data),
        # backfill from company mapper to avoid dropping the whole file.
        if "company" in df.columns:
            missing_mask = df["ticker"].isna()
            if bool(missing_mask.any()):
                mapped = df.loc[missing_mask, "company"].astype(str).apply(mapper.lookup)
                df.loc[missing_mask, "ticker"] = mapped
    else:
        if "company" not in df.columns:
            LOGGER.warning("Missing both ticker and company in %s; skipping.", csv_path)
            return
        df["ticker"] = df["company"].astype(str).apply(mapper.lookup)

    # Fallback 1: historical company->ticker lexicon mined from existing processed outputs.
    if historical_lexicon and company_norm is not None:
        missing_mask = df["ticker"].isna()
        if bool(missing_mask.any()):
            df.loc[missing_mask, "ticker"] = company_norm.loc[missing_mask].map(historical_lexicon)

    # Fallback 2: if company itself looks like a ticker and is in a known symbol universe, use it.
    if enable_short_ticker_fallback and valid_short_tickers and "company" in df.columns:
        missing_mask = df["ticker"].isna()
        if bool(missing_mask.any()):
            short_candidates = (
                df.loc[missing_mask, "company"]
                .astype(str)
                .str.strip()
                .str.upper()
                .str.replace(".", "-", regex=False)
            )
            valid_mask = short_candidates.str.fullmatch(r"[A-Z]{1,5}(?:-[A-Z])?") & short_candidates.isin(valid_short_tickers)
            df.loc[missing_mask, "ticker"] = short_candidates.where(valid_mask)

    unresolved_count = int(df["ticker"].isna().sum())
    if unresolved_count > 0 and "company" in df.columns:
        top_unresolved = (
            df.loc[df["ticker"].isna(), "company"].astype(str).value_counts().head(5).to_dict()
        )
        LOGGER.info(
            "Unresolved ticker mappings in %s: %d / %d (top=%s)",
            csv_path.name,
            unresolved_count,
            len(df),
            top_unresolved,
        )

    df = df.dropna(subset=["published_at", "ticker"]).reset_index(drop=True)
    if df.empty:
        LOGGER.info("No usable rows after ticker filtering for %s", csv_path)
        return
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["yf_ticker"] = df["ticker"].apply(sanitize_ticker)

    # Kept for CLI compatibility. We now expose fixed 5/20 factor windows.
    if price_days != 5:
        LOGGER.warning("--price-days=%s is ignored in factor mode; using fixed windows 5/20.", price_days)

    for col in MARKET_FACTOR_COLS:
        df[col] = pd.NA

    tickers = sorted(df["yf_ticker"].unique())
    min_ts = df["published_at"].min()
    max_ts = df["published_at"].max()
    # Need enough warm-up for 20-day factors and sparse ticker histories.
    start = (min_ts - BDay(80)).date()
    end = (max_ts + BDay(2)).date()
    market_panel = download_market_panel(tickers, start, end, chunk_size)
    if not market_panel:
        LOGGER.warning("No price data fetched for %s; skipping.", csv_path)
        return

    enriched_df = append_market_factors(df, market_panel)
    if enriched_df.empty:
        LOGGER.warning("No rows retained after factor generation for %s", csv_path)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched_df.to_csv(output_path, index=False)
    LOGGER.info("Wrote %s rows -> %s", len(enriched_df), output_path)


def main() -> None:
    configure_logging()
    args = parse_args()

    input_root = Path(args.input)
    emb_root = Path(args.embeddings)
    output_root = Path(args.output)

    overrides: Dict[str, str | None] = {
        "s&p 500": None,
        "sp 500": None,
        "german dax index": None,
        "dax": None,
        "nikkei": None,
        "hang seng": None,
        "dow jones": None,
        "nasdaq": None,
        "walgreens": None,
        "walgreens boots alliance": None,
        "berkshire hathaway": "BRK-B",
        "brk.b": "BRK-B",
        "paramount": None,
        "paramount global": None,
        "coca cola": "KO",
        "under armour": "UAA",
        "mcdonald s": "MCD",
        "home depot": "HD",
        "lowe s": "LOW",
        "target": "TGT",
        "tesla": "TSLA",
        "apple": "AAPL",
        "nvidia": "NVDA",
        "nike": "NKE",
        # Common short/brand aliases frequently seen in transcript summaries.
        "amazon": "AMZN",
        "google": "GOOGL",
        "alphabet": "GOOGL",
        "meta": "META",
        "facebook": "META",
        "palantir": "PLTR",
        "amd": "AMD",
        "alibaba": "BABA",
        "disney": "DIS",
        "costco": "COST",
        "sofi": "SOFI",
        "jp morgan": "JPM",
        "jpmorgan": "JPM",
        "ford": "F",
        "shopify": "SHOP",
        "coinbase": "COIN",
        "gamestop": "GME",
        "robinhood": "HOOD",
        "uber": "UBER",
        "twitter": None,
        "openai": None,
        "youtube": None,
        "cnbc": None,
        "bloomberg": None,
        "federal reserve": None,
    }
    file_overrides = load_alias_overrides(Path(args.alias_overrides))
    overrides.update(file_overrides)
    mapper = CompanyTickerMapper(Path(args.ticker_reference), manual_overrides=overrides)
    LOGGER.info("Total manual overrides active: %d", len(overrides))

    historical_lexicon: Dict[str, str] = {}
    if not args.disable_historical_lexicon:
        historical_lexicon = build_historical_lexicon(
            root=Path(args.historical_lexicon_root),
            min_support=max(1, args.historical_lexicon_min_support),
            excluded_prefixes=[input_root],
        )

    valid_short_tickers: Set[str] = set()
    if not args.disable_short_ticker_fallback:
        valid_short_tickers.update(mapper._mapping.values())  # noqa: SLF001
        vocab_paths = [Path(item.strip()) for item in args.ticker_vocab_paths.split(",") if item.strip()]
        valid_short_tickers.update(load_ticker_vocab_symbols(vocab_paths))
        LOGGER.info("Short-ticker universe size: %d", len(valid_short_tickers))

    channels = None
    if args.channels:
        channels = [item.strip() for item in args.channels.split(",") if item.strip()]

    csv_files = sorted(path for path in input_root.rglob("*.csv") if path.is_file())
    if not csv_files:
        LOGGER.warning("No CSV files found under %s", input_root)
        return
    LOGGER.info("Discovered %d CSV files under %s", len(csv_files), input_root)

    for split_file in csv_files:
        rel = split_file.relative_to(input_root)
        if channels and rel.parts and rel.parts[0] not in channels:
            continue
        emb_path = emb_root / rel.with_suffix(".pt")
        output_path = output_root / rel
        process_file(
            csv_path=split_file,
            emb_path=emb_path,
            output_path=output_path,
            mapper=mapper,
            historical_lexicon=historical_lexicon,
            valid_short_tickers=valid_short_tickers,
            enable_short_ticker_fallback=(not args.disable_short_ticker_fallback),
            price_days=args.price_days,
            chunk_size=args.chunk_size,
        )


if __name__ == "__main__":
    main()
