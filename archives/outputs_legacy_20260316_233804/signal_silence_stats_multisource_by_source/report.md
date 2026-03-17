# Signal vs Silence Report

- replay_root: `data/multisource_ready_22-25/08_replay_buffer_by_source`
- files: `66`
- signal_key: `baseline_actions`
- signal_threshold: `1e-08`

## Global
- |D_sig| = 34478
- |D_sil| = 730336
- |D_sil| / |D_sig| = 21.182667

## Ticker Distribution
- #tickers = 1322
- pct(|D_sil^(i)| > |D_sig^(i)|) = 0.994705
- mean ratio (finite) = 106.043455
- median ratio (finite) = 68.000000
- p25 / p75 (finite) = 30.909091 / 134.500000

## Output Files
- `overall_counts.csv`
- `per_kol_counts.csv`
- `per_kol_split_counts.csv`
- `per_ticker_counts.csv`
- `per_file_counts.csv`
- `distribution_summary.json`
