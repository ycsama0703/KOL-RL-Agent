from autofigure import AutoFigureAgent, Config


prompt = """
Create a professional academic methodology overview diagram for a machine learning / quantitative finance paper.

Title of the figure:
Intent-Preserving Strategy Construction from Financial KOL Discourse

Overall style:
- Top-tier conference paper figure
- Clean, modular, publication-quality
- Landscape layout
- Balanced spacing
- Minimal but elegant icons
- Soft academic color palette
- Distinct semantic color groups:
  - Blue for KOL discourse / intent
  - Orange for missing execution / completion
  - Green for executable portfolio policy
  - Gray for training pipeline / optimization
- Use thin arrows, clean rounded boxes, subtle separators
- Do NOT make it look like software architecture or product workflow
- It should look like a conceptual-methodological framework figure

Main figure structure:
Design the figure as a 4-stage horizontal pipeline with one central conceptual bridge and one bottom training strip.

==================================================
STAGE 1 — RAW KOL DISCOURSE AND INTENT EXTRACTION
==================================================

Place this block on the far left.

Include two input sources:
1. YouTube KOL videos
2. X / social media posts

Merge them into:
- Transcript / text stream

Then connect to a processing module:
- LLM-based Intent Extraction

Inside or under this extraction module, include the extracted fields:
- timestamp
- ticker
- sentiment
- confidence

Output of this stage:
- Structured KOL Signals

Represent this output as a compact table-like or tuple-style block:
(date, ticker, sentiment, confidence)

Use blue tones for all modules in this stage.

==================================================
STAGE 2 — INTENT-ALIGNED BASELINE POLICY
==================================================

Place this block in the middle-left.

Main title of the block:
Intent-Aligned Baseline Policy

This block should show that KOL discourse is first operationalized into a baseline policy anchor.

Inside this block, include two submodules:

2.1 Signal Construction
- sentiment/confidence scoring
- directional aggregation
- daily alignment
- baseline signal generation

2.2 Portfolio Layer / Baseline Allocation
- continuity
- hold decay
- normalization
- position cap

Output of this stage:
- baseline action
- baseline portfolio weights

Label this output explicitly as:
Baseline Action  a_t^{base}

Add a conceptual annotation near this block:
"Expressed by KOL: directional / entry intent"

Use blue-to-teal tones here.

Include the following equation explicitly in this stage in clean LaTeX-style mathematical typography:
\pi^{KOL} : s_t \mapsto a_t^{dir}

This equation should be visually attached to the baseline / intent block, not floating randomly.

==================================================
CENTRAL CONCEPTUAL BRIDGE — PARTIAL TRADING POLICY
==================================================

Place a conceptual bridge block between Stage 2 and Stage 3.
This block is very important and should be visually distinct, perhaps with a thinner outlined box or a highlighted conceptual panel.

Title:
Partial Trading Policy

Inside this block, decompose the trading action into two parts:

Top half:
- Expressed Intent
- asset preference
- directional / entry signal

Bottom half:
- Missing Execution Decisions
- position sizing
- holding duration
- reduction / exit timing

Place the following formula prominently inside this conceptual bridge:
a_t = (a_t^{dir}, a_t^{exec})

Add a small annotation:
"KOL discourse specifies only part of the trading action"

This block should visually communicate that KOL discourse gives a partial policy, not a complete executable strategy.

Use blue/orange mixed styling to show the transition from intent to missing execution.

==================================================
STAGE 3 — INTENT-PRESERVING POLICY COMPLETION
==================================================

Place this block in the middle-right and make it the largest / most important module.

Main title:
Intent-Preserving Policy Completion

Inputs to this stage:
- market features
- baseline action
- last position
- silence duration
- optional state embeddings

Within this stage, create three layers:

3.1 State Representation
A compact input fusion block that merges:
- market state
- KOL baseline
- previous position
- silence features

3.2 Dual-Mode Completion
Split this into two parallel sub-branches:

LEFT BRANCH:
Signal Mode (with KOL signal)
- activated when baseline signal exists
- refine baseline magnitude
- preserve KOL direction
- no reversal

Include a small equation here:
a_t = a_t^{base} + \delta_t^{sig}

RIGHT BRANCH:
Silence Mode (no KOL signal)
- activated during silence periods
- learn decay / reduction / exit
- state-aware execution completion

Optionally include a small equation here:
a_t = g(last\ position,\ market\ state,\ silence)

3.3 Merge Layer
Merge the two branches into:
Constrained Residual Policy

Then output:
Policy Action  a_t^{\pi}

Place the following equation clearly inside this stage:
a_t^{\pi} = a_t^{base} + \delta_t

This equation should visually summarize the main methodological idea.

Use orange tones in this stage.

==================================================
OUTER CONSTRAINT LAYER AROUND STAGE 3
==================================================

Wrap Stage 3 with a dashed or semi-transparent outer box labeled:

Intent Constraints

Split this outer constraint layer into two parts:

Training-time Soft Constraints:
- baseline alignment
- no unsupported entry penalty
- no reversal penalty
- fidelity regularization

Inference-time Hard Constraints:
- no new entry when baseline absent
- no reversal against KOL direction

This outer layer should make it visually obvious that RL is not unrestricted policy optimization.
It is constrained by KOL intent.

==================================================
STAGE 4 — EXECUTABLE PORTFOLIO POLICY
==================================================

Place this block on the far right.

Main title:
Executable Portfolio Policy

Show that policy action goes through a final allocation / execution layer:
- portfolio allocation
- normalization
- exposure cap
- feasibility constraints

Output:
- final portfolio weights
- executable daily strategy

Final output block text:
Unified Trading Policy
(Intent-preserving + executable)

Use green tones here.

==================================================
BOTTOM TRAINING STRIP — OFFLINE TRAINING PIPELINE
==================================================

Place a horizontal training strip across the bottom of the whole figure in gray tones.
This strip should be visually secondary but clearly connected to the main framework.

Title:
Offline Training Pipeline

Show the following flow:

Historical market data
+ Structured KOL signals
+ Baseline action
→ Replay Buffer Construction
→ Behavior Cloning Warm Start
→ Offline RL Optimization
→ Trained Constrained Actor

Within the replay buffer block, mention:
- state
- baseline action
- next state
- reward
- done

Within the BC block, mention:
- fit baseline-aligned behavior

Within the offline RL block, mention:
- value / critic update
- actor refinement
- fidelity-aware optimization

At the right end of the training strip, add a small objective box:
Training Objectives
- reward maximization
- intent preservation
- execution completion

==================================================
VISUAL / TYPOGRAPHIC REQUIREMENTS
==================================================

- Use clean LaTeX-like equation rendering for all formulas
- Equations must be embedded into the relevant modules, not floating as separate annotations
- Avoid overcrowding
- Keep all text concise and readable
- Maintain clear visual hierarchy
- Emphasize that:
  1. KOL discourse defines intent anchors
  2. baseline policy operationalizes expressed intent
  3. missing execution decisions are completed, not replaced
  4. signal and silence are handled differently
  5. the final policy is executable and intent-preserving

Do NOT include:
- low-level code details
- dataloader / optimizer internals
- excessive formulas
- cluttered icons
- decorative illustrations

The figure should look like a high-quality methodology figure for a top ML / multimedia conference paper.

"""

# 1. Configure
config = Config(
    generation_api_key="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",
    generation_provider="openrouter",
    generation_model="google/gemini-3.1-pro-preview",

    enhancement_api_key="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",
    enhancement_provider="openrouter",
    enhancement_model="google/gemini-3.1-pro-preview",  # 和你当前模型一致
    enhancement_input_type="code2prompt",
    enhancement_count=3,
)

# 2. Generate
agent = AutoFigureAgent(config)
result = agent.generate(
    description=prompt,
    max_iterations=5,
    output_format="svg",
    topic="paper",
    enable_enhancement=True,
    enhancement_count=3,
    enhancement_input_type="code2prompt",
    art_style="Modern scientific illustration with clean lines",
)

print(result.svg_path, result.final_score)
print(result.enhanced_paths)