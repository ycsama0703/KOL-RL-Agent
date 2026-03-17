# Signal vs Silence Report

- replay_root: `data/replay_buffer/22-24`
- files: `14`
- signal_key: `actions`
- signal_threshold: `1e-08`

## Global
- |D_sig| = 8569
- |D_sil| = 123552
- |D_sil| / |D_sig| = 14.418485

## Ticker Distribution
- #tickers = 337
- pct(|D_sil^(i)| > |D_sig^(i)|) = 0.997033
- mean ratio (finite) = 51.100284
- median ratio (finite) = 33.333333
- p25 / p75 (finite) = 17.392473 / 62.800000

## Output Files
- `overall_counts.csv`
- `per_kol_counts.csv`
- `per_kol_split_counts.csv`
- `per_ticker_counts.csv`
- `per_file_counts.csv`
- `distribution_summary.json`
