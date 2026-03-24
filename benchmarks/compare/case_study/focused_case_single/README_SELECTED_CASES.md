# Selected Case Studies (Sentiment-Action Aligned)

This folder contains two focused case-study figures under the new plan:

- one figure for **KICL vs Baseline**
- one figure for **KICL vs Ablation (WO_HARD)**

Selection rule used:

- divergence nodes are kept only when evidence can support sentiment-action consistency
- we avoid contradictory narrative patterns (e.g., clearly positive sentiment while the policy exits)

## Case A: KICL vs Baseline

- Case: `x / Jake__Wujastyk`
- Figure:
  - `baseline/x/Jake__Wujastyk/case_single_contrast.png`
  - `baseline/x/Jake__Wujastyk/case_single_contrast.pdf`
- Node (single, strong divergence):
  - `2024-12-05` (mapped event day `2024-12-02`)
- Evidence table:
  - `baseline/x/Jake__Wujastyk/case_single_evidence_aligned.csv`
- Alignment check:
  - aligned evidence rows available (`sentiment_action_aligned = 1`)

## Case B: KICL vs Ablation (WO_HARD)

- Case: `youtube / The_Maverick_of_Wall_Street`
- Comparator: `WO_HARD`
- Figure:
  - `variant/youtube/The_Maverick_of_Wall_Street/case_single_contrast.png`
  - `variant/youtube/The_Maverick_of_Wall_Street/case_single_contrast.pdf`
- Nodes (turning-point focused, shifted left from local peaks):
  - `2024-11-21` (mapped event day `2024-11-11`)
  - `2025-02-18` (mapped event day `2025-02-18`)
  - `2025-03-11` (mapped event day `2025-03-11`)
- Evidence table:
  - `variant/youtube/The_Maverick_of_Wall_Street/case_single_evidence_aligned.csv`
- Alignment check:
  - aligned evidence rows available (`sentiment_action_aligned = 1`)

## Notes

- We intentionally keep only one high-quality node per figure (less but stronger).
- This keeps the narrative focused on structural divergence points rather than parallel-move periods.
