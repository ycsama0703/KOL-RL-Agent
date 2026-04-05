# Public Release Audit

This note summarizes the current audit result for `paper_code_release/`.

## 1. What Was Simplified

The public release has been reduced to the method-facing components only.

### Removed from the public package

- internal training-configuration notes tied to local experiment management
- CUDA-specific pinned wheels and unused dependencies from `requirements.txt`
- any references to local machine paths or internal experiment folder names

### Kept in the public package

- core KICL training entry (`train.py`)
- baseline-construction logic
- replay-buffer construction logic
- evaluation logic for event and daily protocols
- model definitions, data loading, portfolio layer, and ticker embedding utilities
- method documentation needed to map code to the paper

## 2. Files That Are Necessary

The following files are still worth keeping because they define the public method interface.

- `train.py`
- `scripts/add_baseline_action.py`
- `scripts/build_replay_buffer.py`
- `scripts/evaluate_run.py`
- `src/training/models.py`
- `src/training/data.py`
- `src/pipeline/replay_utils.py`
- `src/portfolio/layer.py`
- `src/evaluation/analyzer.py`
- `src/state/ticker_embedding.py`
- `src/utils/logger.py`

These files are not dataset-collection scripts. They define the algorithm, the baseline anchor, the replay representation, and the evaluation protocol.

## 3. Files That Are Lightweight but Fine to Keep

These files are small and harmless, and they help the release behave like a clean standalone repository.

- `README.md`
- `requirements.txt`
- `.gitignore`
- `src/evaluation/__init__.py`
- `src/pipeline/__init__.py`
- `docs/CODE_RELEASE_SCOPE.md`
- `docs/MARKET_FEATURES_AND_BASELINE_DEFINITION.md`
- `docs/SUPPLEMENT_BASELINE_CONSTRUCTION.md`

## 4. Audit Checks Performed

The current public package has been checked for the following:

- no Chinese comments or docstrings remain
- no local absolute paths remain
- no old experiment-root names remain
- no unused heavy dependencies remain in `requirements.txt`
- Python syntax check passes for the released code files

## 5. Current Dependency Scope

The public release now depends only on:

- `torch`
- `numpy`
- `pandas`
- `tqdm`
- `matplotlib`
- `yfinance`

This is a much smaller and cleaner dependency surface than the earlier internal environment.

## 6. Recommended Public Boundary

This release is now suitable as a paper-facing repository if you want to publish:

- core method code
- training code
- evaluation code
- baseline/replay definitions
- explanatory documentation

while still withholding:

- raw data
- data acquisition scripts
- private preprocessing scripts
- proprietary replay buffers / embeddings / outputs

## 7. Final Recommendation

At this point, the package is already close to the minimum reasonable public release.
Further deletion is not strongly recommended, because the remaining files are part of the method definition rather than the private data pipeline.
