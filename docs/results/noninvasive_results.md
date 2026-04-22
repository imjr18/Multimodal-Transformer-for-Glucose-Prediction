# Non-Invasive Results

The committed non-invasive results are currently **smoke-run artifacts**, not a full convergence report. They are still useful because they prove the pipeline executes end to end and they expose the current strengths and weaknesses of the system.

## Baseline Comparison

Source: `noninvasive_glucose/results/baseline_comparison.csv`

| Model | RMSE (mg/dL) | MAE (mg/dL) | Zone A+B | Note |
|------|-------------:|------------:|---------:|------|
| noninvasive_transformer | 21.8088 | 20.9873 | 100.0% | Current-glucose estimate from biosignals only |
| population_mean | 26.4792 | 26.4580 | 100.0% | Predict training-set mean glucose for every window |

Observed gap versus the population-mean baseline:

- absolute RMSE improvement: `4.6703 mg/dL`

## Uncertainty Metrics

Source: `noninvasive_glucose/results/uncertainty_metrics.json`

| Metric | Value |
|--------|------:|
| coverage_95_pct | 100.0% |
| sharpness_mg_dl | 125.35 mg/dL |
| mean_interval_width_fasting | 126.57 mg/dL |

Interpretation:

- the model is **underconfident** in the current smoke run
- `100%` coverage is not a success when it comes with very wide intervals
- the probabilistic plumbing works, but the current weights are not yet calibrated enough for deployment-quality uncertainty

## Fine-Tuning History

Source: `noninvasive_glucose/results/finetune_history.json`

Current committed history contains one smoke-run epoch:

| Epoch | Train Loss | Val Loss | Val RMSE | Val MAE | LR |
|------:|-----------:|---------:|---------:|--------:|---:|
| 1 | 0.4300 | 0.3743 | 19.7931 | 19.7632 | 0.0000045 |

Interpretation:

- this is enough to show the NLL objective is numerically stable
- this is **not** enough to claim final model performance

## Calibration Demo

Source: committed repository narrative in `noninvasive_glucose/NONINVASIVE_LIVING_EXPLANATION.md`

Held-out user slice:

- before calibration: `22.1598 mg/dL`
- after calibration: `18.5515 mg/dL`
- absolute improvement: `3.6084 mg/dL`
- relative improvement: `16.28%`

Interpretation:

- even in the smoke setup, updating only the user embedding moves the model in the right direction
- this supports the calibration design, even though the exact magnitude should be re-measured after a full run

## Attribution Summary

Source: `noninvasive_glucose/results/noninvasive_attribution_summary.csv`

### Fasting / stable

- HR: `65.74%`
- ECG: `9.47%`
- EMG: `5.06%`
- EEG: `13.87%`
- CBF: `5.86%`

### Deep sleep

- HR: `62.05%`
- ECG: `9.75%`
- EMG: `4.78%`
- EEG: `16.41%`
- CBF: `7.01%`

Interpretation:

- HR currently dominates the committed smoke attributions
- EEG contribution rises in `deep_sleep`, which is directionally sensible
- the modality balance has not fully matured yet, which is consistent with the model being undertrained

## Honest Summary

The current non-invasive project has three important committed facts:

1. It beats a trivial population-mean baseline by about `4.67 mg/dL` RMSE.
2. Calibration improved one held-out slice by `3.61 mg/dL` in the smoke demo.
3. Uncertainty is currently too wide to be clinically useful, which is exactly the kind of failure a smoke artifact should reveal before stronger claims are made.

