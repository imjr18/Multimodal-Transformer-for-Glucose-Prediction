# Supervised Results

This document reports the **actual committed artifact status** of the supervised project. The codebase is complete, but the repository snapshot does not preserve the final experiment outputs for Parts A-E.

## Summary Table

| Part | Expected summary artifact(s) | Committed status | What is still known from the repo |
|------|------------------------------|------------------|-----------------------------------|
| Part A | `part_a/checkpoints/model_comparison.json`, training history, attention figures | Missing | `TemporalTransformer` exists and has `112,834` trainable parameters |
| Part B | `part_b/results/fusion_comparison.csv`, `ablation_results.csv` | Missing | Model sizes are recoverable from code: early `104,962`, late `313,346`, cross `338,754` |
| Part C | `part_c/results/benchmark.csv` | Missing | Full-model parameter counts are recoverable from code; `frequency_eeg` is the Part D default |
| Part D | `part_d/results/cross_archetype_results.csv`, adaptation curve figure | Missing | FOMAML and adaptation evaluation are implemented |
| Part E | `part_e/results/ig_scenario_summary.csv`, `spurious_correlation.json`, final report | Missing | Full interpretability stack is implemented |

## Part A

Implemented evaluation:

- RMSE at 30 and 60 minutes
- MAE at 30 and 60 minutes
- Clarke Error Grid zone summaries
- LSTM baseline comparison

Committed artifact status:

- no final Part A metric JSON is present in the repository snapshot

Concrete model fact that is recoverable:

- `TemporalTransformer` parameter count: `112,834`

## Part B

Implemented evaluation:

- fusion-strategy comparison
- modality ablation
- cross-attention visualisation

Committed artifact status:

- `fusion_comparison.csv` is not present
- `ablation_results.csv` is not present

Concrete model facts that are recoverable:

- `EarlyFusionTransformer`: `104,962` parameters
- `LateFusionTransformer`: `313,346` parameters
- `CrossModalTransformer`: `338,754` parameters

## Part C

Implemented evaluation:

- benchmark over EEG encoders
- peak VRAM logging
- latency timing
- sleep-stage attention analysis

Committed artifact status:

- `benchmark.csv` is not present

Concrete model facts that are recoverable:

- `FullModalTransformer` with `frequency_eeg`: `610,178` parameters
- `FullModalTransformer` with `patch_eeg`: `613,954` parameters
- `FullModalTransformer` with `hierarchical_eeg`: `626,754` parameters

Operational clue:

- Part D is configured to carry `frequency_eeg` forward as the default EEG backbone

## Part D

Implemented evaluation:

- adaptation curves at multiple support set sizes
- cross-archetype RMSE breakdown
- embedding-space visualisation

Committed artifact status:

- no `cross_archetype_results.csv`
- no adaptation curve figure
- no final Part D summary JSON

Interpretation:

- the personalisation logic is implemented, but the repository snapshot does not preserve the final population-level metrics needed for publication claims

## Part E

Implemented evaluation:

- attention rollout
- integrated gradients scenario analysis
- probing classifiers
- spurious-noise control
- head specialisation
- final report generation

Committed artifact status:

- no committed Part E result CSV/JSON outputs in this snapshot

Interpretation:

- the interpretability tooling is present and runnable, but this repository snapshot does not ship the resulting evidence bundle

## Bottom Line

For the supervised project, the important publication-ready task is **artifact preservation**, not missing implementation work. Before claiming final Part A-E numbers in a paper or GitHub release, rerun the stages and commit or archive the small summary outputs:

- final metric JSON/CSV files
- compact benchmark tables
- compact interpretability summaries

Without those, the code can be audited, but the empirical story cannot be fully quoted.

