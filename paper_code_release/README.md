# Paper Code Release

This directory is a **method-only public release** prepared for the paper.

It includes the core code needed to understand and reproduce:

- KOL-aligned baseline construction
- replay-buffer fields and transition logic
- dual-branch residual policy training
- BC -> IQL optimization
- hard intent constraints
- event/daily evaluation and betrayal metrics

It intentionally **does not** include:

- raw X / YouTube data
- data collection scripts
- data cleaning / mapping / aggregation scripts tied to the proprietary dataset
- private intermediate datasets, embeddings, or annotations
- plotting / paper-production utilities that are not part of the core method


## Directory Layout

- `train.py`
  - Main KICL training entry.
  - Contains the BC stage, IQL stage, dual-branch residual actor logic, hard intent constraints, and metric computation.

- `scripts/add_baseline_action.py`
  - Constructs the baseline raw score from sentiment/confidence.

- `scripts/build_replay_buffer.py`
  - Builds replay buffers from structured samples.
  - Defines stored transition fields and metadata.

- `scripts/evaluate_run.py`
  - Unified evaluation entry for event-level and daily evaluation.
  - Computes betrayal metrics and baseline comparisons.

- `src/training/models.py`
  - Actor / critic / value network definitions.

- `src/training/data.py`
  - Replay dataset loader and dataloader helpers.

- `src/pipeline/replay_utils.py`
  - Shared replay-building logic:
    - state construction
    - baseline weight construction
    - behavior weight smoothing
    - portfolio reward construction

- `src/portfolio/layer.py`
  - Portfolio allocation layer that maps ticker-level raw scores into executable portfolio weights.

- `src/evaluation/analyzer.py`
  - Policy replay and position logging utilities used in evaluation.

- `src/state/ticker_embedding.py`
  - Ticker embedding loader/encoder used during state construction.

- `src/utils/logger.py`
  - Logging utility used by training/evaluation entry points.

- `docs/`
  - Supplementary notes describing:
    - market factor exposure
    - baseline construction
    - public release scope


## What This Release Is Meant To Support

This release is designed to let readers reconstruct the **methodological core** of the paper:

1. How a KOL-aligned executable baseline is constructed
2. How structured samples are turned into replay transitions
3. How the residual policy is trained around the baseline anchor
4. How hard admissibility constraints are applied
5. How the resulting policy is evaluated against the baseline

It is **not** designed to reproduce the proprietary dataset pipeline end-to-end.


## Minimal External Inputs Expected

To run this code on another dataset, the user must provide their own structured inputs with fields compatible with the replay-building logic, including at least:

- ticker
- published_at
- sentiment
- confidence
- baseline_raw_score
- reward_1d
- text embedding columns (or a compatible substitute)

In practice, this release should be viewed as:

- **public method code**
- plus **public interface definition**
- but **without the proprietary data layer**


## Recommended Citation Description

If you need one sentence in the supplementary material:

> We publicly release the core training, inference, baseline-construction, replay-buffer, and evaluation code for our intent-preserving policy completion framework, while withholding the proprietary data acquisition and preprocessing pipeline associated with the high-cost KOL dataset.


## Suggested Public Repo Boundary

If this directory is later moved into a standalone public repository, the recommended boundary is:

- keep everything currently inside `paper_code_release/`
- do not add `data/`
- do not add dataset-specific preprocessing scripts
- optionally add a tiny toy example buffer schema for API demonstration
