# Code Release Scope

This note documents the intended public-release boundary for the paper code.


## 1. Publicly Released

The following components are part of the methodological contribution and should be public:

### 1.1 Core training logic

- `train.py`
- `src/training/models.py`
- `src/training/data.py`
- `src/utils/logger.py`

These files define the model architecture, BC -> IQL training flow, residual action parameterization, and optimization details.


### 1.2 Baseline construction and replay representation

- `scripts/add_baseline_action.py`
- `scripts/build_replay_buffer.py`
- `src/pipeline/replay_utils.py`
- `src/portfolio/layer.py`
- `src/state/ticker_embedding.py`

These files define:

- how KOL-derived sentiment/confidence are mapped into baseline anchor actions
- how behavior actions are derived
- how replay states, actions, rewards, and transitions are constructed
- how raw per-ticker scores are converted into executable portfolio weights


### 1.3 Evaluation logic

- `scripts/evaluate_run.py`
- `src/evaluation/analyzer.py`

These files define:

- event-level evaluation
- daily evaluation
- betrayal metrics
- baseline vs policy comparison


### 1.4 Method documentation

- `docs/MARKET_FEATURES_AND_BASELINE_DEFINITION.md`
- `docs/SUPPLEMENT_BASELINE_CONSTRUCTION.md`
- `docs/CODE_RELEASE_SCOPE.md`

These files make the implementation-to-paper mapping explicit.


## 2. Intentionally Withheld

The following components are tied to the proprietary dataset and are not part of the public release:

### 2.1 Raw and intermediate data

- raw social-media data
- transcription outputs
- daily aggregated KOL files
- replay buffers derived from the proprietary dataset
- embeddings generated from the proprietary text corpus


### 2.2 Data collection and proprietary preprocessing

- platform-specific crawling / downloading scripts
- entity mapping scripts tied to the proprietary KOL corpus
- cleaning and transformation scripts whose sole purpose is to reproduce the private dataset
- dataset-specific analysis scripts that expose the proprietary corpus structure


## 3. Release Principle

The public release follows this rule:

> release the code needed to reproduce the **method**, but not the assets needed to reconstruct the **proprietary dataset**.

Concretely:

- **method layer**: public
- **training/evaluation layer**: public
- **data acquisition / dataset construction layer**: private


## 4. Why This Boundary Is Defensible

This boundary preserves scientific reproducibility for the algorithmic contribution because readers can still inspect:

- the baseline definition
- the state/action/reward formulation
- the residual policy design
- the hard-constraint mechanism
- the BC/IQL optimization logic
- the evaluation metrics and benchmark protocol

At the same time, it protects the high-cost data asset used in the paper.


## 5. Practical Recommendation

If this code is later uploaded to a public paper repository, the repository should contain:

- all files in this `paper_code_release/` directory
- a short top-level README
- a minimal toy-format example or schema description

and should exclude:

- `data/`
- private benchmark outputs
- dataset build scripts
- private analysis artifacts
