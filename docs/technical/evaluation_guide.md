# Evaluation Guide

This guide explains where each project writes metrics, what those metrics mean, and which outputs are currently committed in the repository.

## Supervised project evaluation flow

### Part A

Evaluation logic lives in:

- `glucose_transformer/part_a/evaluate.py`
- `glucose_transformer/run_part_a.py`

Metrics produced:

- RMSE at `30` and `60` minutes
- MAE at `30` and `60` minutes
- Clarke Error Grid zones for both horizons

Expected saved artifacts:

- `part_a/checkpoints/training_history.json`
- `part_a/checkpoints/model_comparison.json`
- attention figures under `part_a/figures/`

Committed status:

- not present in this snapshot

### Part B

Evaluation logic lives in:

- `glucose_transformer/part_b/evaluate.py`
- `glucose_transformer/part_b/ablation.py`
- `glucose_transformer/run_part_b.py`

Metrics produced:

- RMSE and MAE at `30` and `60` minutes
- mean Zone A+B percentage
- per-model inference time
- modality ablation deltas

Expected saved artifacts:

- `part_b/results/fusion_comparison.csv`
- `part_b/results/ablation_results.csv`
- `part_b/results/synthetic_feature_validation.json`

Committed status:

- not present in this snapshot

### Part C

Evaluation logic lives in:

- `glucose_transformer/part_c/benchmark.py`
- `glucose_transformer/run_part_c.py`

Metrics produced:

- RMSE at `30` and `60` minutes for each EEG variant
- peak VRAM
- average inference time
- trainable parameter count

Expected saved artifact:

- `part_c/results/benchmark.csv`

Committed status:

- not present in this snapshot

### Part D

Evaluation logic lives in:

- `glucose_transformer/part_d/evaluate.py`
- `glucose_transformer/run_part_d.py`

Metrics produced:

- adaptation RMSE at `0`, `5`, `15`, `30`, and `60` minutes of support data
- cross-archetype RMSE summary
- percent improvement after adaptation

Expected saved artifacts:

- `part_d/results/cross_archetype_results.csv`
- `part_d/figures/adaptation_curve.png`
- `part_d/figures/embedding_space.png`

Committed status:

- not present in this snapshot

### Part E

Evaluation logic lives in:

- `glucose_transformer/part_e/attention_rollout.py`
- `glucose_transformer/part_e/integrated_gradients.py`
- `glucose_transformer/part_e/probing_classifiers.py`
- `glucose_transformer/part_e/spurious_correlation.py`
- `glucose_transformer/part_e/head_specialisation.py`

Expected saved artifacts:

- `part_e/results/ig_scenario_summary.csv`
- `part_e/results/spurious_correlation.json`
- `part_e/FINAL_REPORT.md`

Committed status:

- not present in this snapshot

## Non-invasive project evaluation flow

Evaluation logic lives in:

- `noninvasive_glucose/evaluate/metrics.py`
- `noninvasive_glucose/evaluate/uncertainty_eval.py`
- `noninvasive_glucose/evaluate/compare_baselines.py`
- `run_noninvasive.py`

Committed artifacts present now:

- `noninvasive_glucose/results/baseline_comparison.csv`
- `noninvasive_glucose/results/uncertainty_metrics.json`
- `noninvasive_glucose/results/noninvasive_attribution_summary.csv`
- `noninvasive_glucose/results/finetune_history.json`

What they currently show:

- smoke-run RMSE of `21.81 mg/dL` for the non-invasive model
- baseline comparison against a population-mean predictor
- `100%` nominal 95% coverage with very wide intervals, which indicates underconfident smoke-run uncertainty
- two attribution scenarios currently surfaced in the committed results: `fasting_stable` and `deep_sleep`

## Reproducing metrics cleanly

1. Delete or ignore old generated outputs if you want a fresh run.
2. Run the relevant stage entry point.
3. Confirm that the expected CSV/JSON/PNG files were written.
4. Record the commit hash alongside any reported numbers if you plan to publish them.

## Publication caution

The supervised repository is code-complete but does not currently ship its final numeric artifacts. The non-invasive repository ships smoke artifacts, not convergence artifacts. For a publication-ready release, rerun the missing stages and preserve the resulting summary files before making strong empirical claims.

