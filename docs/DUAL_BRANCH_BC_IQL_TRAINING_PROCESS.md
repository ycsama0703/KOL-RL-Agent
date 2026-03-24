# Dual-Branch BC+IQL Training Process (Implementation-Aligned)

This note summarizes the current training implementation in this repo:
- `train.py`
- `src/training/models.py`
- `src/pipeline/replay_utils.py`
- `scripts/augment_with_market_data.py`
- `scripts/build_replay_buffer.py`

It is written to match the actual code path, not an abstract variant.

Compared to a vanilla offline BC/IQL pipeline, this implementation is intentionally
**baseline-anchored, residualized, regime-aware, and constraint-aware**. The design
goal is not to maximize return with unrestricted policy freedom, but to improve
execution quality while preserving KOL intent boundaries.

---

## 1) What is fed into training (state/action/reward)

The training sample is not a plain `(state, action, reward)` tuple from market-only
features. Instead, each transition carries an explicit **intent anchor** (`baseline_action`)
plus execution context (`last_position`, `silence_days`) and compact market context.
This is the core reason the later dual-branch completion can be interpreted as
“intent completion” rather than “intent replacement”.

## 1.1 State construction

Each sample state is built as:

\[
s_t = [e_t^{text} \,\|\, e_t^{ticker} \,\|\, x_t^{core} \,\|\, x_t^{mkt}]
\]

where:

- \(e_t^{text}\): text embedding vector
- \(e_t^{ticker}\): ticker embedding vector
- \(x_t^{core}\): core scalar features
  - `sentiment`
  - `confidence`
  - `last_position`
  - `silence_days`
- \(x_t^{mkt}\): 6 compact market factors
  - `ret_1d`
  - `ret_5d`
  - `vol_5d`
  - `vol_20d`
  - `volu_z_20d`
  - `dist_sma20`

Code references:
- Market factors are generated in `scripts/augment_with_market_data.py`
- State concatenation is in `src/pipeline/replay_utils.py::build_states`

## 1.2 Baseline and behavior actions

Replay samples contain two action-related signals:

- `baseline_action` (\(a_t^{base}\)): KOL-intent anchor
- `action` (\(a_t^{beh}\)): behavior action used to train critic/value and actor fit term

Code references:
- Buffer packing: `scripts/build_replay_buffer.py`
- Dataset loading: `src/training/data.py::ReplayDataset`

## 1.3 Reward used in training

Training uses `portfolio_rewards` when available (preferred), otherwise falls back to per-row reward.

Code reference:
- `src/training/data.py::ReplayDataset` (`rewards_tensor = data.get("portfolio_rewards", data["rewards"])`)

---

## 2) Dual-branch policy and residual formulation

Instead of directly predicting a free action, the actor predicts a **residual**
around the KOL baseline. This reparameterization gives a stable center
(\(a_t^{base}\)) and lets the model focus on missing execution details
(sizing/decay adjustments) with lower risk of semantic drift.

## 2.1 Actor structure

The actor is dual-head:

- shared backbone
- `head_signal` \(\rightarrow \delta_t^{sig}\)
- `head_decay` \(\rightarrow \delta_t^{sil}\)

Code reference:
- `src/training/models.py::ActorNetwork`

## 2.2 Regime routing (signal vs silence)

Routing is based on baseline magnitude:

- signal regime when \(|a_t^{base}|\) is not near zero
- silence regime otherwise

Then select the corresponding residual head:

\[
\delta_t =
\begin{cases}
\delta_t^{sig}, & |a_t^{base}| > \epsilon \\
\delta_t^{sil}, & |a_t^{base}| \le \epsilon
\end{cases}
\]

Code reference:
- `train.py::_extract_delta`

## 2.3 Residual action composition

Policy is residual:

\[
a_t^\pi = a_t^{base} + \delta_t
\]

with residual clamp:

\[
\delta_t \leftarrow \text{clip}(\delta_t, -c, c)
\]

Code reference:
- `train.py::apply_intent_constraints`

---

## 3) Hard constraints and soft penalties

Hard rules define admissible action space; soft penalties shape optimization inside
that space. In practice, this prevents degenerate behavior where an unconstrained
objective would exploit market-only signals and drift away from KOL semantics.

## 3.1 Hard admissibility constraints (default ON)

When `hard_intent_constraints=True`, two hard rules are enforced:

1. No unsupported new entry in silence regime  
   if \(|a_t^{base}| < \text{entry\_threshold}\), force \(a_t^\pi = 0\)

2. No direction reversal against baseline sign  
   if \(a_t^{base} > 0\), clamp \(a_t^\pi \ge 0\);  
   if \(a_t^{base} < 0\), clamp \(a_t^\pi \le 0\)

Code reference:
- `train.py::apply_intent_constraints`

## 3.2 Soft intent penalties

Even with hard constraints enabled, soft penalties are still computed:

- entry penalty (`entry_pen`)
- reversal penalty (`rev_pen`)

Code reference:
- `train.py::intent_penalties_soft`

---

## 4) Modified BC stage (warm start)

BC is run first for `bc_epochs`.

Per batch:
1. forward actor to get dual-head residual
2. route by regime
3. compose \(a_t^\pi = a_t^{base} + \delta_t\)
4. apply hard constraints (if enabled)
5. optimize BC objective

Default BC mode is `bc_fit_behavior=True`:

\[
\mathcal{L}_{BC}
=
\underbrace{\|a_t^\pi-a_t^{beh}\|_2^2}_{\text{fit behavior}}
+
\lambda_{anchor}\underbrace{\|a_t^\pi-a_t^{base}\|_2^2}_{\text{anchor}}
+
\lambda_{entry}P_{entry}
+
\lambda_{rev}P_{rev}
\]

If `bc_fit_behavior=False`, BC fits baseline directly.

Code reference:
- `train.py::behavior_cloning`

Practical role of this stage:
- It initializes the actor in an intent-consistent region before value-driven updates.
- It reduces early IQL instability on sparse-signal buffers.
- It keeps the residual heads grounded to executable trajectories rather than random
  unconstrained deltas.

---

## 5) Modified IQL stage (fine-tuning)

After BC, IQL runs for `iql_steps`.

## 5.1 Residual-aware value conditioning

Value/critic use extended state:

\[
\tilde{s}_t = [s_t \,\|\, a_t^{base}]
\]

and behavior residual action:

\[
\delta_t^{beh} = a_t^{beh} - a_t^{base}
\]

Code reference:
- `train.py::iql_training`

## 5.2 Critic target with fidelity-shaped reward

Policy action is first built from actor residual (with constraints), then used to compute a fidelity shaping term:

\[
r_t^{aug}
=
r_t
-
\lambda_{fid}\,\|\text{sg}(a_t^\pi)-a_t^{base}\|_2^2
\]

\[
y_t = r_t^{aug} + \gamma(1-d_t)V(\tilde{s}_{t+1})
\]

Critic regression:

\[
\mathcal{L}_Q = \|Q(\tilde{s}_t,\delta_t^{beh}) - y_t\|_2^2
\]

Code reference:
- `train.py::iql_training` (critic section)

## 5.3 Value update (expectile)

\[
\mathcal{L}_V = \text{Expectile}_\tau\left(Q(\tilde{s}_t,\delta_t^{beh}) - V(\tilde{s}_t)\right)
\]

Code reference:
- `train.py::expectile_loss`, `train.py::iql_training` (value section)

## 5.4 Actor update (advantage-weighted behavior fit + intent shaping)

Behavior advantage:

\[
A_t^{beh} = Q(\tilde{s}_t,\delta_t^{beh}) - V(\tilde{s}_t)
\]

Weight:

\[
w_t = \text{clip}\left(\exp(\beta A_t^{beh}),\text{max}=100\right)
\]

Main fit term:

\[
\mathcal{L}_{fit}
=
\mathbb{E}\left[w_t\|a_t^\pi-a_t^{beh}\|_2^2\right]
\]

Plus alignment and soft penalties:

\[
\mathcal{L}_{actor}
=
\mathcal{L}_{fit}
+
\lambda_{align}\|a_t^\pi-a_t^{base}\|_2^2
+
\lambda_{entry}P_{entry}
+
\lambda_{rev}P_{rev}
\]

Code reference:
- `train.py::iql_training` (actor section)

Practical role of this stage:
- Critic/value learn offline return structure on behavior trajectories.
- Actor remains behavior-compatible via advantage-weighted regression, but is still
  tethered to baseline semantics through alignment and intent penalties.
- Fidelity-shaped reward discourages “winning by drifting too far from anchor”.

---

## 6) Final policy composition (what is deployed/evaluated)

At inference/evaluation time, the same structure is used:

1. get dual residual heads
2. route by regime (if split enabled)
3. compose residual action with baseline anchor
4. enforce hard admissibility constraints

So the deployed policy is:

\[
\pi(s_t, a_t^{base}) \rightarrow a_t^\pi
\]

with **baseline-anchored residual completion**, not free-form action generation.

Code references:
- `train.py::_extract_delta`
- `train.py::apply_intent_constraints`
- `train.py::evaluate`

---

## 7) Practical interpretation

This implementation operationalizes:

- **Anchor**: KOL baseline action is always present
- **Dual-branch completion**:
  - signal head refines active signals
  - silence/decay head handles no-fresh-signal regime
- **Constrained optimization**:
  - hard admissibility to prevent explicit betrayal
  - soft regularization to stabilize and shape learning
- **BC + IQL coupling**:
  - BC provides stable, intent-aware warm start
  - IQL improves execution quality under offline data

In short: the model learns to **complete** KOL intent into a tradable policy, rather than replace the intent source.

---

## 8) End-to-end training narrative (for method figure/text)

The full training loop can be described as:

1. Build enriched event samples with text/ticker embeddings, execution context, and
   compact market factors.
2. Build replay transitions with both `baseline_action` (intent anchor) and
   `behavior action` (execution target for offline value learning).
3. Initialize a dual-head residual actor by BC warm start under the same hard
   admissibility constraints used at evaluation.
4. Run IQL on extended states \([s_t \| a_t^{base}]\), using behavior residuals for
   critic/value updates.
5. Update actor by advantage-weighted regression to behavior action, while adding
   baseline alignment + entry/reversal penalties.
6. Deploy the same regime routing + residual merge + hard constraints at inference.

So the final policy is explicitly:
\[
\text{KOL anchor} + \text{regime-conditioned residual completion}
\]
which is a constrained completion policy, not a fully free actor.

---

## 9) What is new in our method (vs standard BC/IQL)

Relative to a standard offline BC/IQL baseline, our method introduces the following
implementation-level changes:

1. **Anchor-conditioned policy parameterization**
   - Standard: actor outputs action directly.
   - Ours: actor outputs residual and composes with baseline action
     (\(a_t^\pi = a_t^{base} + \delta_t\)).

2. **Dual-branch regime split**
   - Standard: single actor head for all states.
   - Ours: `signal` / `silence` heads with routing based on baseline activation.

3. **Hard admissibility constraints in policy construction**
   - Standard: no strict semantic action filter.
   - Ours: no unsupported entry in silence regime + no reversal against baseline direction.

4. **Intent-aware regularization during BC and IQL actor update**
   - Baseline alignment term
   - Soft entry penalty
   - Soft reversal penalty

5. **Fidelity-shaped reward for critic target**
   - Augment reward with baseline-deviation penalty to reduce gains obtained purely
     by semantic drift.

6. **Residual-aware critic/value conditioning**
   - Train value functions on extended state \([s \| a^{base}]\), making value
     estimation explicitly anchor-aware.

These changes jointly enforce the principle:
**improve execution flexibility without breaking KOL-intent admissibility**.

---

## 10) BC/IQL + Our Modifications (Figure-Ready Unified View)

This section fuses the standard BC/IQL backbone and our method-specific modules
into one single training graph, so the method figure can be drawn naturally.

## 10.1 Unified pipeline (single mainline)

\[
\text{Inputs} \rightarrow \text{Dual-Branch Residual Actor} \rightarrow \text{BC Warm Start} \rightarrow \text{IQL Fine-tuning} \rightarrow \text{Final Policy}
\]

Where each stage contains a "standard core" plus "our constraints/anchor design":

1. **Inputs (anchor-aware state)**
   - Standard core: state features, offline transitions
   - Our part: baseline anchor \(a_t^{base}\) is explicitly carried with each sample;
     market factors are compact (6-dim) and auxiliary.

2. **Dual-Branch Residual Actor**
   - Standard core: actor network
   - Our part:
     - residual action form \(a_t^\pi = a_t^{base} + \delta_t\)
     - signal/silence heads (`delta_signal`, `delta_decay`)
     - regime routing by baseline activation

3. **BC Warm Start**
   - Standard core: supervised behavior fitting
   - Our part:
     - baseline anchor penalty (\(\|a^\pi-a^{base}\|^2\))
     - soft entry/reversal penalties
     - hard admissibility already applied during BC forward action
     - optional mode switch (`bc_fit_behavior`)

4. **IQL Fine-tuning**
   - Standard core:
     - critic TD regression
     - value expectile regression
     - actor advantage-weighted regression
   - Our part:
     - residual-aware conditioning with extended state \([s \| a^{base}]\)
     - behavior residual action \((a^{beh}-a^{base})\) for Q/V learning
     - fidelity-shaped reward for critic target
     - actor-side baseline alignment + soft intent penalties
     - policy action always constructed through anchor+residual, then constrained

5. **Final Policy**
   - Standard core: deterministic action output
   - Our part:
     - same regime routing as training
     - same hard admissibility at inference
     - final output is constrained completion, not free-form trading action

## 10.2 Suggested figure layering (directly drawable)

For a clean method diagram, use 3 layers:

- **Layer A (Backbone / standard BC+IQL):**
  - Offline Replay Data
  - BC Warm Start
  - IQL (Critic / Value / Actor)
  - Final Policy

- **Layer B (Our architectural changes):**
  - Baseline Anchor Input
  - Dual-Branch Residual Actor
  - Regime Gating (signal vs silence)
  - Residual Merge \(a^{base}+\delta\)

- **Layer C (Our constraint-shaping changes):**
  - Hard Admissibility (no unsupported entry, no reversal)
  - Soft Intent Penalties (entry/reversal)
  - Baseline Alignment
  - Fidelity Reward Shaping

Interpretation for readers:
- Layer A explains "it is still BC+IQL".
- Layer B explains "why it is intent-completion instead of a free actor".
- Layer C explains "how optimization is prevented from drifting off intent".

## 10.3 One-sentence caption-ready summary

“Our method keeps the standard BC→IQL offline learning backbone, but injects
baseline-anchored residual dual-branch policy construction and intent-preserving
hard/soft constraints throughout both stages, yielding a constrained completion
policy rather than an unconstrained trading actor.”
