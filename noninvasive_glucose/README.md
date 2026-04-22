# Non-Invasive Glucose Estimation

Estimating current blood glucose from biosignals alone — no CGM required at inference time. The model outputs both a point estimate and uncertainty, and it supports low-shot calibration from three finger-prick readings.

## Key Difference from the Supervised Project

The supervised project predicts future glucose with recent glucose history available as input. This project removes glucose from the inference path entirely. Glucose is used only during training as a teacher signal and during sparse onboarding as a calibration target. That change forces three architectural shifts: a shorter 30-minute window, a probabilistic output head, and a deployment-time calibration loop that updates only the user embedding.

## How It Works

Each biosignal modality gets its own compact Transformer encoder: HR, ECG-derived HRV, EMG envelope features, EEG band powers, and CBF. These modality-specific token streams are fused with HR-centric cross-attention, because autonomic state is the most stable hub signal across the available wearable channels. A final encoder processes the fused HR tokens together with a user-conditioning token.

The output head is `UncertaintyHead`, not a deterministic regression layer. It produces a mean glucose estimate and a log-variance, and training uses Gaussian negative log-likelihood rather than MSE. At inference time, Monte Carlo Dropout keeps dropout active and samples the model repeatedly so the system can estimate epistemic uncertainty in addition to the learned aleatoric variance.

## Calibration

Calibration is handled by `UserCalibrator` in `calibration/calibrate.py`. The user supplies three strategically spaced reference readings:

- fasting / stable
- post-meal peak
- post-exercise recovery

The model then deep-copies itself, freezes every encoder and fusion weight, and updates only the 16-dimensional user embedding through a short SGD inner loop. The result is a personalised version of the same shared physiological model, not a separate per-user network.

## Uncertainty

Uncertainty is estimated with Monte Carlo Dropout plus the predicted variance from the uncertainty head. Practically, the model returns:

- a mean glucose estimate
- an epistemic standard deviation from repeated stochastic forward passes
- an aleatoric standard deviation from the learned variance head
- a total predictive standard deviation and derived confidence interval

This matters because non-invasive glucose estimation is intrinsically ambiguous in some windows. A useful system must know when the evidence is weak, not just output a number.

## File Structure

```text
noninvasive_glucose/
├── NONINVASIVE_EXPLANATION.md
├── NONINVASIVE_LIVING_EXPLANATION.md
├── config.py
├── simulation/              # Synthetic cohort + calibration-session generation
├── models/                  # Signal encoders, fusion, uncertainty head, full model
├── training/                # Pretraining and full-model fine-tuning
├── calibration/             # Deployment-time adaptation logic
├── evaluate/                # Metrics, uncertainty evaluation, baseline comparison
├── interpretability/        # Integrated gradients without glucose context
└── data/README_DATA.md      # Real-data adapter notes + synthetic fallback
```

## Results

The currently committed results are from a smoke run rather than a full convergence run. They are still useful for verifying that the pipeline executes end to end.

| Metric | Value | Notes |
|--------|------:|-------|
| RMSE (uncalibrated) | 21.81 mg/dL | `baseline_comparison.csv`, non-invasive model row |
| RMSE (after calibration) | 18.55 mg/dL | Small held-out calibration demo from `NONINVASIVE_LIVING_EXPLANATION.md` |
| Coverage (95% CI) | 100.0% | Underconfident smoke-run intervals; not deployment-grade yet |
| Zone A+B | 100.0% | From current smoke artifact only |

## Accuracy vs Supervised Model

The non-invasive project currently reports a committed smoke-run RMSE of `21.81 mg/dL`. The repository is set up to compare this directly against a supervised reference artifact, but that supervised metric file is not present in the committed snapshot, so the exact measured gap is not yet materialised here. See [docs/results/comparison_table.md](../docs/results/comparison_table.md) for the honest side-by-side status.

## Theory

Full theoretical explanation: [Non-invasive overview](../docs/theory/07_noninvasive_overview.md)
