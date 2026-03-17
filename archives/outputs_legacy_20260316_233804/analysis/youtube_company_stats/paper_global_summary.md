# YouTube Global Statistics (Paper-Ready)

Source directory: `outputs/analysis/youtube_company_stats`

## Dataset

- Rows: 48722

- KOLs: 18

- Companies: 6774

- Date range: 2022-01-01 15:45:00+00:00 to 2025-12-31 22:01:17+00:00

## Ticker/Company Mention Frequency

- Companies total: 6774.0

- Mentioned once: 4204 (62.06%)

- Mentioned >=2 times: 2570 (37.94%)

- Mentions per company (mean/median/p90/max): 7.19 / 1.00 / 9.00 / 1491.00

## Silence Duration (days between consecutive mentions)

- All: median=5.64, p90=107.13, max=1425.34, n=41948

- Positive-only: median=7.14, p90=129.09, max=1409.29, n=22094

- Negative-only: median=13.06, p90=185.04, max=1352.84, n=10750

## Sentiment Reversal

- Eligible companies (>=2 directional mentions): 2228

- Companies with reversal: 1297

- Reversal rate: 58.21%

- Time-to-first-reversal (median/p90/max days): 179.84 / 753.91 / 1403.23

## Signal Imbalance

- Positive/Negative/Neutral ratio: 54.25% / 27.70% / 18.05%

- P(next=negative | current=positive): 20.91%

- P(next=positive | current=negative): 41.57%

## Long Silence After First Mention

- No follow-up within 30 days: 6068 (89.58%)

- No follow-up within 60 days: 5809 (85.75%)

- No follow-up within 90 days: 5614 (82.88%)
