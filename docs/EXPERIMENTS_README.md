# Experiments README

This README specifies how to present the experimental results for the paper. The goal is not to maximize the number of tables and figures, but to organize the evidence around the paper’s central claim:

> We open a practical path for constructing executable strategies from KOL intent by using only minimal market information, while keeping the learned policy aligned with the original KOL intent instead of letting market signals overwhelm it.

The experimental presentation should therefore emphasize two things simultaneously:

1. **Strategy performance**
2. **Intent preservation / betrayal control**

It should **not** be framed primarily as “our method gets the highest return.”

---

## 1. Core experimental narrative

All result presentation should support the following narrative:

- Signal-only methods tend to be more faithful to KOL intent, but are too rigid and incomplete as executable strategies.
- Unconstrained or weakly constrained learning methods may improve returns or Sharpe, but often do so by drifting away from KOL intent.
- Our method improves flexibility and executability while keeping betrayal of KOL intent substantially lower.
- The broader message is not that our current method is the final answer, but that **there is real headroom for intent-preserving KOL-guided strategy completion**, even when using only a very small set of simple market features.

---

## 2. Section structure for the paper

Use the following subsection structure in the paper:

```latex
\section{Experiments}

\subsection{Experimental Setup}

\subsection{Evaluation Metrics}

\subsection{Results}

\subsection{Ablation Study}

\subsection{Case Study}
```

Keep the subsection titles simple. Put the methodological specificity into the content rather than the headings.

---

## 3. Data usage strategy in experiments

There are effectively three data scopes, but this does **not** need to be explicitly presented as a “three-level hierarchy” in the paper.

### 3.1 Large KOL corpus
Use this only to support:
- the discourse statistics in Problem Formulation
- broad empirical observations about sparsity, silence, delayed reversals, etc.
- assumptions about why unconstrained RL can drift away from KOL intent

This should mostly stay in earlier sections and not be repeated heavily in the experiments section.

### 3.2 Selected KOL subset for quantitative experiments
Use the selected high-impact KOL subset for:
- training
- benchmarking
- ablations
- main quantitative results

This is the main dataset for the experiments section.

### 3.3 Representative KOL cases
Use 1–2 representative KOLs for case studies only.

Recommended:
- one **YouTube** KOL case
- one **X** KOL case

This helps show that the method works across both long-form and short-form financial discourse settings.

---

## 4. What must be shown in the main paper

The main paper should include:

### Required tables
1. **Main results table**
2. **Ablation table**

### Required figures
1. **Method overview figure** (already exists)
2. **At least one cross-method tradeoff figure**
3. **Two case-study figures** (recommended: one YouTube, one X)

Everything else can go to the appendix.

---

## 5. Evaluation philosophy

The evaluation must be explicitly framed as **dual-objective**:

### A. Conventional trading quality
Standard finance metrics such as:
- cumulative return
- Sharpe ratio
- maximum drawdown (MDD)

### B. Intent preservation / betrayal control
This is the key innovation in evaluation.
Use the betrayal metrics already defined in the paper:
- **Unsupported Entry Rate (UER)**
- **Direction Reversal Rate (DRR)**
- **Baseline Deviation (BD)**

The experiments should show that some methods may improve conventional performance only by becoming less KOL-like and more market-driven.

---

## 6. Main results table design

### Table name
Suggested title:
- **Main comparison of strategy performance and KOL-intent preservation**

### Recommended columns
Use a compact set of metrics. Do not overload the table.

Recommended columns:
- Method
- Return ↑
- Sharpe ↑
- MDD ↓
- UER ↓
- DRR ↓
- BD ↓

If space is too tight, MDD can be moved to the appendix, but ideally it stays.

### Recommended row grouping
Group rows by method family:

#### Signal-based baselines
Examples:
- simple KOL signal strategy
- fixed holding rule
- sentiment-to-position rule
- baseline-only strategy

#### Unconstrained / weakly constrained learning baselines
Examples:
- market-driven RL
- text-as-feature RL
- no-constraint completion model

#### Ours
- full method

### What the table should communicate
The table should make it easy to read the following pattern:

- signal-only methods: low betrayal, weak flexibility
- unconstrained learning methods: stronger market adaptation, but high betrayal
- ours: improved strategy performance with substantially lower betrayal than unconstrained alternatives

### Formatting recommendation
Bold only the best result in each column, and maybe underline the second-best if the venue style permits.
Do **not** over-format.

---

## 7. Cross-platform presentation (YouTube vs X)

Because the experiments use both YouTube and X, the paper should include **one explicit cross-platform comparison**, but this should not explode the main tables.

### Best option
Use a **figure** rather than a second large table.

### Recommended figure
A scatter plot or two-panel comparison showing:
- x-axis: betrayal score (or one betrayal metric such as BD / combined betrayal score)
- y-axis: Sharpe or Return
- marker color: platform (YouTube vs X)
- marker shape: method

This is a compact way to show:
- cross-platform consistency
- performance–intent tradeoff
- where the proposed method sits relative to baselines

### If a table is preferred instead
Make a compact platform-split table with only a few columns, e.g.:
- Sharpe
- UER
- DRR

for YouTube and X separately.

But the **figure option is strongly preferred** because it is visually clearer and more compact.

---

## 8. Ablation table design

### Table name
Suggested title:
- **Ablation study on policy completion design**

### Recommended rows
- Full model
- w/o fidelity
- w/o no-reversal
- w/o no-entry
- w/o signal/silence split
- w/o baseline anchoring (if available)

### Recommended columns
Keep the same columns as the main table if possible:
- Return ↑
- Sharpe ↑
- UER ↓
- DRR ↓
- BD ↓

If space is tight, drop MDD first, not the betrayal metrics.

### What the table should communicate
Each ablation should demonstrate that removing a design component causes one of the following:
- more unsupported entry
- more direction reversal
- larger deviation from the KOL-aligned baseline
- poorer balance between fidelity and performance

This table should prove that the method is not just a generic RL model with arbitrary decorations.

---

## 9. Which figures to show in the main paper

### Figure A: Tradeoff figure
**Strongly recommended.**

Purpose:
- show that the proposed method occupies a better region in the performance–betrayal tradeoff space

Recommended design:
- x-axis: betrayal score or BD
- y-axis: Sharpe or Return
- different markers for different methods
- color by platform or family

This figure is especially helpful because the paper’s contribution is not just “better performance,” but “better performance without large betrayal.”

---

### Figure B: Case study for a YouTube KOL
Purpose:
- illustrate one representative long-form discourse case
- show how the model behaves under active KOL guidance and later silence

Recommended contents:
- price curve in the background
- timestamps of KOL mentions
- short translated KOL text snippets or event labels
- baseline action / position curve
- final positions of:
  - ours
  - one signal-only baseline
  - one unconstrained baseline

What to emphasize:
- ours remains aligned with KOL direction
- ours still adapts execution behavior
- unconstrained baselines may reverse or create unsupported positions

---

### Figure C: Case study for an X KOL
Purpose:
- illustrate one representative short-form social posting case
- complement the YouTube example and demonstrate cross-platform relevance

Recommended contents:
Same format as the YouTube case study.

What to emphasize:
- the method works even when discourse is shorter and more fragmented
- silence handling is especially important in this regime

---

## 10. Case study layout recommendation

Each case-study figure should ideally contain:

1. **Price curve**
2. **KOL event markers**
3. **Baseline action / baseline position**
4. **Method position curves**
   - ours
   - one signal-only baseline
   - one unconstrained or weakly constrained model

Optional annotations:
- “active bullish signal”
- “no new signal”
- “unconstrained reversal”
- “ours reduces without reversing”

Do **not** make the figure too text-heavy.
A few concise annotations are enough.

---

## 11. What should go to the appendix

Move the following to the appendix unless space is abundant:

- platform-split full tables
- per-KOL detailed results
- more than two case studies
- robustness checks with additional metrics
- additional traditional financial metrics beyond Return / Sharpe / MDD
- full definitions and formulas of evaluation metrics (already planned)

The appendix should serve as breadth; the main paper should prioritize clarity.

---

## 12. Ranking of importance

If page budget becomes tight, keep the following in the main paper in this priority order:

1. Main results table
2. Ablation table
3. One tradeoff figure
4. Two case-study figures (or one combined case-study figure if necessary)
5. Additional platform-specific detail → appendix

---

## 13. What not to do

Do **not** do the following:

- do not flood the main paper with per-KOL tables
- do not use too many conventional finance metrics in the main results table
- do not make the experiments read like a generic return-maximization trading paper
- do not let case studies become too decorative or anecdotal
- do not focus the narrative on “we achieved the best return”

---

## 14. Final message the experiments should support

All experimental presentation should reinforce the following interpretation:

> The contribution of the paper is not simply that the current model produces slightly better returns. Rather, it demonstrates that there is meaningful room to learn execution flexibility around KOL intent, even when only a minimal set of basic market features is used. This opens a practical path for future work to explore more advanced intent-preserving strategy construction methods.

---

## 15. Suggested deliverables for Codex

Please help generate and refine the following:

1. **Main results table** with compact formatting and grouped method rows
2. **Ablation table** with the same betrayal-focused metric columns
3. **One tradeoff figure** (performance vs betrayal)
4. **Two case-study figures** (YouTube and X)
5. **Appendix tables** for additional per-platform / per-KOL results
6. Optional: a **combined case-study figure** if page budget is tight

The final presentation should optimize for:
- clarity
- argumentative strength
- intent-preservation emphasis
- minimal redundancy
