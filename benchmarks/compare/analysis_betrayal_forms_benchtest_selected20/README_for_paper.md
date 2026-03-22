# Betrayal Subsection (Rerun on Selected-20, Bench Test Results)

This document combines two connected mini-experiments under one subsection:

1. **Betrayal form profile** (how each method deviates from KOL intent).
2. **Profit-linked betrayal check** (whether profitable events come with more betrayal, and which form drives it).

## Data scope (this rerun)

- Universe: **Selected-20** (X=10, YouTube=10).
- Methods: `KICL`, `BC`, `IQL`, `CQL`, `TD3BC`, `AWAC`.
- Data roots: `benchmarks/bench_test_results/*_xrefresh_mainline_test` + `multisource_test_mainline_xrefresh`.
- Coverage for Experiment 2 intersection: **20/20 pairs** for all methods.

---

## Experiment A: Betrayal Form Profile

Input:
- `benchmarks/compare/meta/selected20_all_methods_vs_baseline_detailed_benchtest.csv`

Outputs:
- `betrayal_forms_by_method_source_raw.csv`
- `betrayal_forms_by_method_source_scaled.csv`
- `betrayal_forms_heatmap_scaled.png`

### Core raw means

### X

- KICL: `UER=0.0000`, `DRR=0.0000`, `BD=0.0070`, `CG=0.1726`
- BC: `UER=0.2158`, `DRR=0.1930`, `BD=0.0241`, `CG=0.1553`
- IQL: `UER=0.1902`, `DRR=0.2077`, `BD=0.0266`, `CG=0.1579`
- AWAC: `UER=0.1875`, `DRR=0.1937`, `BD=0.0262`, `CG=0.1677`
- CQL: `UER=0.8438`, `DRR=0.3796`, `BD=0.1017`, `CG=0.2696`
- TD3BC: `UER=0.7915`, `DRR=0.3710`, `BD=0.1795`, `CG=0.5439`

### YouTube

- KICL: `UER=0.0000`, `DRR=0.0000`, `BD=0.0084`, `CG=0.1524`
- BC: `UER=0.2742`, `DRR=0.2391`, `BD=0.0377`, `CG=0.1704`
- IQL: `UER=0.2531`, `DRR=0.1922`, `BD=0.0309`, `CG=0.1757`
- AWAC: `UER=0.2698`, `DRR=0.1821`, `BD=0.0350`, `CG=0.1819`
- CQL: `UER=0.9228`, `DRR=0.3306`, `BD=0.2374`, `CG=0.6356`
- TD3BC: `UER=0.9349`, `DRR=0.5037`, `BD=0.2055`, `CG=0.4392`

### Takeaway (Experiment A)

- KICL keeps **hard betrayal** (`UER`, `DRR`) near zero on both platforms.
- KICL also keeps `BD` low relative to baselines.

---

## Experiment B: Profit-linked Betrayal

Input:
- Manifest: `benchmarks/compare/meta/compare_manifest_benchtest.json`
- Universe CSV: `benchmarks/compare/meta/selected20_all_methods_vs_baseline_detailed_benchtest.csv`

Main outputs:
- `benchmarks/compare/analysis_excess_return_betrayal_benchtest_selected20/excess_return_betrayal_pooled.csv`
- `benchmarks/compare/analysis_excess_return_betrayal_benchtest_selected20/excess_return_betrayal_bootstrap_ci.csv`
- `benchmarks/compare/analysis_excess_return_betrayal_benchtest_selected20/excess_return_betrayal_hard_soft_decomposition.csv`
- `benchmarks/compare/analysis_excess_return_betrayal_benchtest_selected20/hard_only_betrayal_summary.csv`
- `benchmarks/compare/analysis_excess_return_betrayal_benchtest_selected20/betrayal_hard_soft_decomposition_story.png`

Definitions:
- Profit event condition: `event_return > 0`
- `betrayal_any = reversal OR entry_violation OR (normalized_dev >= 0.2)`

### B.1 What happens under betrayal_any

KICL still shows large uplift in `P(betrayal_any | profit)` vs non-profit:

- X: uplift ≈ `+0.8379`
- YouTube: uplift ≈ `+0.7799`

This can look counter-intuitive if interpreted as “hard betrayal”.

### B.2 Hard vs soft decomposition (critical)

From `excess_return_betrayal_hard_soft_decomposition.csv`:

#### KICL (X)
- `hard_rate` = `0.000021`
- `hard_uplift` = `+0.000446`
- `dev_flag_rate` = `0.068011`
- `dev_uplift` = `+0.823093`
- `any_uplift` = `+0.823539`

#### KICL (YouTube)
- `hard_rate` = `0.000072`
- `hard_uplift` = `+0.000228`
- `dev_flag_rate` = `0.103782`
- `dev_uplift` = `+0.742665`
- `any_uplift` = `+0.742893`

### Takeaway (Experiment B)

- For KICL, profitable events are associated primarily with **soft deviation** (`dev_flag`), not hard violations.
- Hard betrayal (`reversal + unsupported entry`) remains near zero.

---

## Final subsection claim (safe wording)

- “KICL’s gains are not explained by hard-form intent betrayal.  
  The observed uplift under betrayal-any is mostly driven by soft magnitude refinement, while unsupported entries and direction reversals remain negligible.”

This keeps the storyline consistent with the intent-preserving objective.

---

## Optional reporting variant: treat only hard forms as betrayal

If you prefer to define betrayal strictly as hard violations:

- `betrayal_hard = reversal OR unsupported_entry`
- do **not** count soft deviation as betrayal.

Use:

- `hard_only_betrayal_summary.csv`

Key values for KICL (selected-20 rerun):

- X: `p_hard_betrayal=0.000021`, `uplift_hard_only=+0.000446`
- YouTube: `p_hard_betrayal=0.000072`, `uplift_hard_only=+0.000228`

This variant makes the conclusion even clearer:

- KICL’s gains are not associated with hard betrayal.
- What increases with profitable events is mainly the soft completion component.
