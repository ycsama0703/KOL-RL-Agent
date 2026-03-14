"""Demo script: augment first event with latest 5-day prices using yfinance."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
import yfinance as yf
from pandas.tseries.offsets import BDay


TRAIN_PATH = Path("data/processed/cleaned/Everything_Money/train.csv")
EMB_PATH = Path("data/processed/embeddings/Everything_Money/train.pt")
OUTPUT_PATH = Path("data/processed/cleaned/Everything_Money/demo_first_event_with_prices.csv")

# Minimal name->ticker map for the demo KOL sample.
COMPANY_TICKER_MAP: Dict[str, str] = {
    "nike": "NKE",
    "walgreens": "WBA",
    "walgreens boots alliance": "WBA",
    "starbucks": "SBUX",
    "coca-cola": "KO",
    "under armour": "UAA",
}


def fetch_last_five_closes(ticker: str, event_date: pd.Timestamp) -> Optional[List[float]]:
    """Fetch the most recent 5 business-day close prices up to event_date."""
    event_ts = pd.Timestamp(event_date)
    if event_ts.tzinfo is None:
        event_ts = event_ts.tz_localize("UTC")
    event_ts = event_ts.tz_convert(None)
    start_date = (event_ts - BDay(10)).date()
    end_date = (event_ts + pd.Timedelta(days=1)).date()

    try:
        history = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:  # pragma: no cover - network failure
        print(f"[WARN] Failed to download {ticker}: {exc}")
        return None
    if history.empty:
        return None

    if isinstance(history.columns, pd.MultiIndex):
        if ("Close", ticker) not in history.columns:
            return None
        closes = history[("Close", ticker)]
    else:
        if "Close" not in history.columns:
            return None
        closes = history["Close"]

    closes = closes.dropna().tail(5)
    if len(closes) < 5:
        return None
    return closes.to_list()


def main() -> None:
    df = pd.read_csv(TRAIN_PATH)
    if df.empty:
        raise SystemExit("train.csv is empty; nothing to process.")

    if not EMB_PATH.exists():
        raise SystemExit(f"{EMB_PATH} not found; please generate embeddings first.")
    payload = torch.load(EMB_PATH, map_location="cpu")
    embeddings = payload["embeddings"]
    if len(embeddings) != len(df):
        raise SystemExit("Embedding rows do not match CSV rows; aborting.")

    id_col = "event_id" if "event_id" in df.columns else "video_id"
    first_event_id = df.iloc[0][id_col]
    event_rows = df[df[id_col] == first_event_id].copy()
    publish_ts = pd.to_datetime(event_rows.iloc[0]["published_at"], utc=True)

    enriched_rows = []
    for _, row in event_rows.iterrows():
        company_name = str(row["company"]).strip().lower()
        ticker = COMPANY_TICKER_MAP.get(company_name)
        if not ticker:
            continue
        closes = fetch_last_five_closes(ticker, publish_ts)
        if closes is None:
            continue
        enriched = row.to_dict()
        labels = ["close_t-4", "close_t-3", "close_t-2", "close_t-1", "close_t-0"]
        for label, price in zip(labels, closes):
            enriched[label] = price

        emb_vector = embeddings[row.name]
        if hasattr(emb_vector, "tolist"):
            emb_vector = emb_vector.tolist()
        for idx, value in enumerate(emb_vector):
            enriched[f"embedding_{idx}"] = value
        enriched_rows.append(enriched)

    if not enriched_rows:
        raise SystemExit("No rows could be enriched with price data.")

    out_df = pd.DataFrame(enriched_rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved demo rows with price history -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
