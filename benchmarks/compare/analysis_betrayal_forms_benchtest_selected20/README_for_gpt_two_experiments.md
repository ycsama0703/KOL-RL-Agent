# Betrayal Analysis Subsection (Two Mini-Experiments, Selected-20)

This note is prepared as direct input for GPT-assisted paper writing.
It summarizes the setup, metrics, figures, and key findings for the two linked
mini-experiments in the betrayal-analysis subsection.

---

## 0) Scope and Data Base

- Data base: **Selected-20** (`10 X + 10 YouTube`).
- Methods: `KICL`, `BC`, `IQL`, `CQL`, `TD3BC`, `AWAC`.
- Rerun source roots: `benchmarks/bench_test_results/*`.
- Coverage for Experiment 2 (intersection mode): **20/20 pairs for all methods**.

Core meta files:

- `benchmarks/compare/meta/selected20_all_methods_vs_baseline_detailed_benchtest.csv`
- `benchmarks/compare/meta/compare_manifest_benchtest.json`

---

## 1) Mini-Experiment A: Betrayal Form Profile

### Goal

Identify **which form of betrayal** each method mainly exhibits.

### Metrics (method-level mean by source)

- `UER`: unsupported entry rate
- `DRR`: direction reversal rate
- `BD`: mean absolute deviation
- `CG`: correlation gap (`1 - baseline_policy_corr`)

### Main figure (for paper)

- Heatmap:
  - `benchmarks/compare/analysis_betrayal_forms_benchtest_selected20/betrayal_forms_heatmap_scaled.png`

### Key numeric results (raw means)

From:

- `benchmarks/compare/analysis_betrayal_forms_benchtest_selected20/betrayal_forms_by_method_source_raw.csv`

#### X

- KICL: `UER=0.0000`, `DRR=0.0000`, `BD=0.0070`, `CG=0.1726`
- BC: `UER=0.2158`, `DRR=0.1930`, `BD=0.0241`, `CG=0.1553`
- IQL: `UER=0.1902`, `DRR=0.2077`, `BD=0.0266`, `CG=0.1579`
- AWAC: `UER=0.1875`, `DRR=0.1937`, `BD=0.0262`, `CG=0.1677`
- CQL: `UER=0.8438`, `DRR=0.3796`, `BD=0.1017`, `CG=0.2696`
- TD3BC: `UER=0.7915`, `DRR=0.3710`, `BD=0.1795`, `CG=0.5439`

#### YouTube

- KICL: `UER=0.0000`, `DRR=0.0000`, `BD=0.0084`, `CG=0.1524`
- BC: `UER=0.2742`, `DRR=0.2391`, `BD=0.0377`, `CG=0.1704`
- IQL: `UER=0.2531`, `DRR=0.1922`, `BD=0.0309`, `CG=0.1757`
- AWAC: `UER=0.2698`, `DRR=0.1821`, `BD=0.0350`, `CG=0.1819`
- CQL: `UER=0.9228`, `DRR=0.3306`, `BD=0.2374`, `CG=0.6356`
- TD3BC: `UER=0.9349`, `DRR=0.5037`, `BD=0.2055`, `CG=0.4392`

### Takeaway A

- KICL nearly eliminates hard betrayal forms (`UER`, `DRR`) on both sources.
- KICL also keeps `BD` low.
- Several generic RL baselines (especially `CQL`, `TD3BC`) are hard-betrayal dominated.

---

## 2) Mini-Experiment B: Profit-Linked Betrayal Decomposition

### Goal

Test whether methods’ profitable events are associated with betrayal,
and decompose that association into hard vs soft components.

### Condition and definitions

- Profit-event condition: `event_return > 0`
- Baseline “any betrayal”:
  - `betrayal_any = reversal OR entry_violation OR (normalized_dev >= 0.2)`
- Hard betrayal:
  - `hard = reversal OR entry_violation`
- Soft/non-hard component:
  - residual (mainly deviation component)

### Main figure (for paper)

- Decomposition figure:
  - `benchmarks/compare/analysis_excess_return_betrayal_benchtest_selected20/betrayal_hard_soft_decomposition_story.png`

### Supporting files

- `benchmarks/compare/analysis_excess_return_betrayal_benchtest_selected20/excess_return_betrayal_pooled.csv`
- `benchmarks/compare/analysis_excess_return_betrayal_benchtest_selected20/excess_return_betrayal_bootstrap_ci.csv`
- `benchmarks/compare/analysis_excess_return_betrayal_benchtest_selected20/excess_return_betrayal_hard_soft_decomposition.csv`
- `benchmarks/compare/analysis_excess_return_betrayal_benchtest_selected20/hard_only_betrayal_summary.csv`

### Key findings

#### 2.1 Under betrayal_any, KICL shows strong uplift

- X: `uplift_vs_nonexcess = +0.8379`
- YouTube: `uplift_vs_nonexcess = +0.7799`

(KOL-level bootstrap CI excludes zero in both sources.)

#### 2.2 Decomposition explains why this does NOT mean “hard betrayal for profit”

For KICL:

- X:
  - `hard_rate = 0.000021`
  - `hard_uplift = +0.000446`
  - `dev_uplift = +0.823093`
  - `any_uplift = +0.823539`
- YouTube:
  - `hard_rate = 0.000072`
  - `hard_uplift = +0.000228`
  - `dev_uplift = +0.742665`
  - `any_uplift = +0.742893`

Interpretation:

- KICL’s uplift is almost entirely from **soft deviation / completion**, not hard violations.
- Hard betrayal remains near zero.

For comparison:

- `CQL` / `TD3BC` have very high hard rates, and hard-uplift is strongly negative.
- This indicates hard betrayal is not a reliable source of profitable-event uplift.

### Takeaway B

- If soft deviation is included in betrayal_any, KICL can appear to have high uplift.
- But once separated, KICL’s gain is **not** from hard betrayal.
- This supports the intended narrative: gains come from constrained completion rather than intent-breaking actions.

---

## 3) Writing-ready narrative (concise)

Suggested storyline:

1. Experiment A shows KICL has the cleanest betrayal-form profile (especially hard forms).
2. Experiment B shows profitable-event uplift exists, but decomposition attributes KICL’s uplift to soft completion, not hard betrayal.
3. Therefore, KICL improves performance while preserving core intent constraints.

---

## 4) Optional strict variant

If the paper wants to define betrayal **strictly as hard forms only**,
use:

- `hard_only_betrayal_summary.csv`

In that strict view, KICL hard betrayal and hard-uplift remain near zero on both sources.

