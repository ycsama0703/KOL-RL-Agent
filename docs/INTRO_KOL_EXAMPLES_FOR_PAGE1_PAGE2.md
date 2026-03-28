# Intro Revision Pack: Concrete KOL Statement Examples

This file provides concrete KOL-style statement examples for the intro/early related-work revision.
Goal: make the distinction between **crowd sentiment** and **KOL discourse** immediately intuitive.

## 1) Short Example Candidates (directly usable in a bubble)

Use one of the following as the primary example bubble on page 1:

1. **X / Jake__Wujastyk (2025-09-05)**
   - "ONDS still think this is probably $10+ by end of year."

2. **X / Jake__Wujastyk (2024-12-02)**
   - "A textbook support/resistance flip over the last few days."
   - (paired ticker context in source: SOUN)

3. **YouTube / The_Maverick_of_Wall_Street (2025-04-07)**
   - "The Nasdaq is sinking with massive losses in Tesla, Nvidia, and Apple."

4. **YouTube / The_Maverick_of_Wall_Street (2025-09-10)**
   - "Tesla is range-bound; we need a breakout up or down before acting."

5. **YouTube / Financial_Education (2025-07-22)**
   - "Shopify is an absolute animal; the stock is up over 56% in three months."

6. **YouTube / Financial_Education (2024-11-29)**
   - "Amazon is just always a buy."

## 2) Recommended primary pair for intro

If you want a clean contrast (bullish + risk warning), use:

- Bullish side: "ONDS still think this is probably $10+ by end of year."
- Risk/caution side: "Tesla is range-bound; we need a breakout up or down before acting."

This pair highlights that KOL discourse often carries **directional intent + execution caution**, not a single scalar sentiment.

## 3) Copy-ready paragraph (for intro bridge)

> Consider a typical KOL statement such as "ONDS still think this is probably $10+ by end of year," or "Tesla is range-bound; we need a breakout before acting."  
> These expressions are not generic crowd mood indicators. They provide **asset-specific directional guidance** and often include an **implicit execution condition** (e.g., wait for confirmation, control sizing).  
> Therefore, we treat KOL discourse as a weighted, expert-conditioned directional signal rather than plain social-media sentiment.

## 4) Copy-ready sentence for contribution prelude

> If we collapse such KOL discourse into a single sentiment score, we lose the distinction between directional intent and execution under-specification; this is exactly why we model it as a partial policy and perform intent-preserving completion.

## 5) Optional concise bubble texts (very short style)

If the figure space is tight, use one of these:

- "NVDA still looks strong here, but sizing should stay cautious."
- "Still bullish on ONDS; likely $10+ by year-end."
- "TSLA is range-bound; wait for breakout confirmation."

## 6) Case-study selected examples (recommended to add)

These are directly aligned with the two case studies already used in the paper analysis.

### Case A: X / Jake__Wujastyk (baseline vs KICL)

- **Node focus day**: 2024-12-05
- **Mapped event day**: 2024-12-02
- **Primary quote (SOUN, bullish):**
  - "my thesis of a strong upside move coming on friday was extremely simple. a textbook s/r flip over the last few days."
- **Supporting quote (LTC, caution/late-cycle):**
  - "#litecoin are ripping, generally this is closer to the end of the cycle than the beginning."
- **Supporting quote (FXI, conditional upside):**
  - "if the backtests that i have presented this weekend come to fruition this coming week, #china names may be the highest performers. $fxi"

Short bubble candidate for intro:
- "A textbook S/R flip suggests strong upside into Friday."

### Case B: YouTube / The_Maverick_of_Wall_Street (KICL vs WO-HARD)

- **Node 1 focus day**: 2024-11-21
- **Mapped event day**: 2024-11-11
- **Primary quote (macro-tech caution):**
  - "...nvidia up 26% ... plugin amd amd is down 7% in the same period of time..."
- **Primary quote (TSLA execution caution):**
  - "...tesla ... leave alone for a little bit..."

- **Node 2 focus day**: 2025-02-18
- **Mapped event day**: 2025-02-18
- **Primary quote (BABA high-conviction bullish):**
  - "and i want to own the stack to be honest with you folks ..."

Short bubble candidates for intro:
- "AMD is lagging despite the AI run; be selective."
- "I want to own this stack."

## 7) Source files (for traceability)

- `benchmarks/compare/case_study/focused_case_single/baseline/x/Jake__Wujastyk/case_single_evidence_aligned.csv`
- `benchmarks/compare/case_study/focused_case_single/variant/youtube/The_Maverick_of_Wall_Street/case_single_evidence_aligned.csv`
- `benchmarks/compare/case_study/focused_case/x/Jake__Wujastyk/case_dual_node_evidence.csv`
- `benchmarks/compare/case_study/focused_case/youtube/The_Maverick_of_Wall_Street/case_dual_node_evidence.csv`
