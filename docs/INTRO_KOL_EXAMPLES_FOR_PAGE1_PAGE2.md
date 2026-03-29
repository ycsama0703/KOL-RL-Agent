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

## 2.1) Context windows for the selected two examples (for "strategy completion" argument)

Below are the exact local contexts of the two selected statements.
Purpose: show that these are mostly **directional cues / wait conditions**, without full execution instructions.

### A) X / Jake__Wujastyk / 2025-09-05 / ONDS

- Source file: `benchmarks/compare/case_study/trace/x/Jake__Wujastyk/kicl_discourse_evidence.csv`
- Event id: `x_Jake__Wujastyk_2025-09-05`
- Full text in record:
  - `"$onds still think this is probably $10+ by end of year."`

Sentence-level check (same record):
- Previous sentence: *(none)*
- Target sentence: `"$onds still think this is probably $10+ by end of year."`
- Next sentence: *(none)*
- Sentence count in this record: `1`

Interpretation for writing:
- This is a pure directional/high-level conviction cue.
- It does **not** specify execution details (timing rule, sizing, stop, rebalance path).

### B) YouTube / The_Maverick_of_Wall_Street / 2025-09-10 / TSLA

- Source file: `benchmarks/compare/case_study/trace/youtube/The_Maverick_of_Wall_Street/kicl_discourse_evidence.csv`
- Event id: `2RvXpCho-zs`
- Full local paragraph in record:
  1. `anything to see in tesla here, no, it's been rangebound.`
  2. `you can argue that we have higher lows, but static highs.`
  3. `there's nothing for me to do here in tesla.`
  4. `we need a breakout one one way or the other, up or down before we make up a decision, but right now tesla is just not looking attractive at all.`

Target-centered window:
- Previous sentence: `there's nothing for me to do here in tesla.`
- Target sentence: `we need a breakout one one way or the other, up or down before we make up a decision, but right now tesla is just not looking attractive at all.`
- Next sentence: *(none)*

Interpretation for writing:
- This is a wait/condition statement (\"act only after breakout\").
- It still leaves execution under-specified (no explicit allocation, no concrete position trajectory, no risk-budget rule).

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

## 8) Node-near discourse windows (for intro replacement)

This section is aligned to the **current case-study nodes** (do not change case study itself).
Use these windows to replace intro examples so the intro and case-study evidence stay consistent.

### 8.1 X case node neighborhood (Jake__Wujastyk)

- Case node: `focus_day=2024-12-05`, mapped event day `2024-12-02`
- Event id: `x_Jake__Wujastyk_2024-12-02`

Core quote (node-aligned):
- `SOUN`: "My thesis of a strong upside move coming on Friday was extremely simple. A textbook S/R flip over the last few days."

Same-day neighboring cues (immediately around the same event block):
- `LTC`: "#Litecoin are ripping, generally this is closer to the end of the cycle than the beginning."
- `FXI`: "IF the backtests that I have presented this weekend come to fruition this coming week, #China names may be the highest performers. $FXI"
- `ETH`: "#Ethereum Watching for a potentially massive breakout on the horizon. $ETHUSD"

Why this helps intro:
- Asset-specific and directional (SOUN/LTC/FXI/ETH are explicit).
- Contains conviction/condition language.
- Still missing full execution specification (no explicit size, no full exit plan, no portfolio-level rebalance path).

### 8.2 YouTube case node neighborhood (The_Maverick_of_Wall_Street, node #1)

- Case node #1: `focus_day=2024-11-21`, mapped event day `2024-11-11`
- Event id: `EbZZ7d7GrtM`

Core quote (node-near, same event):
- `Tesla`: "...And the last one is Tesla of course is it too hot right now ... don't short it right now ... leave alone for a little bit..."

Adjacent same-event lines (same local block in source order):
- previous: `Advanced Micro Devices`: "...Nvidia up 26% ... AMD is down 7% in the same period of time..."
- next: `Nvidia`: "...last time around the trade war was most negative to semiconductors..."

Additional same-event conditional line:
- `Match Group`: "...you need to see a reversal first ... some buying with higher volume ... it could be a falling knife..."

Why this helps intro:
- Not generic crowd mood; it is asset-level commentary with explicit conditional language.
- Shows "wait / caution / condition first" behavior but remains under-specified for full execution.

### 8.3 YouTube case node neighborhood (The_Maverick_of_Wall_Street, node #2)

- Case node #2: `focus_day=2025-02-18`, mapped event day `2025-02-18`
- Event id: `ZAkC0dBvQyE`

Core quote:
- `Alibaba`: "and I want to own the stack to be honest with you folks ... I might do the spread first ..."

Same-day neighboring market cues (different videos, same mapped day):
- "I'm actually thinking about shortening LEN..."
- "you have another company ... and you're probably going to have many more..."

Why this helps intro:
- High-conviction directional intent exists ("want to own").
- But execution remains partial and open-ended (no complete size-horizon-exit tuple).

### 8.4 Ready-to-use intro pair (node-consistent)

If you want a pair that is fully consistent with current case-study nodes:

1. X/Jake (2024-12-02, SOUN):
   - "A textbook S/R flip over the last few days."
2. YouTube/Maverick (2024-11-11, Tesla block):
   - "Tesla ... leave alone for a little bit ... you need confirmation."

This pair keeps the intro aligned with case-study evidence while still supporting the "partial policy" argument.

## 9) Explicit-in-text candidates (to avoid ticker-mapping ambiguity)

If you want intro examples that are defensible without relying on the `company/ticker` label, use lines where the asset is explicitly present in text.

### 9.1 Node-consistent explicit candidates (recommended)

From current case-study node neighborhood:

- **X / Jake / 2024-12-02 (same case node event)**
  - `"$FXI ... may be the highest performers. $FXI"`
  - `"$UBER Looks ready for launch."`
  - `"#Ethereum ... breakout ... $ETHUSD"`

- **YouTube / Maverick / 2024-11-11 (same case node event)**
  - `"Tesla ... don't short it right now ... leave alone for a little bit ..."`
  - `"AMD is down 7% in the same period of time ..."`

### 9.2 Suggested intro pair (explicit + case-aligned)

Use this pair if you want both "explicit asset mention" and "aligned with current case-study nodes":

1. **X/Jake (2024-12-02)**  
   `"$FXI ... may be the highest performers. $FXI"`
2. **YouTube/Maverick (2024-11-11)**  
   `"Tesla ... leave alone for a little bit ..."`

Note:
- The X quote has explicit ticker symbols (`$FXI`) in-text.
- The YouTube quote has explicit asset-name mention (`Tesla`) in-text.
- This is stronger for intro than the SOUN sentence, which is case-aligned but ticker-implicit in wording.
