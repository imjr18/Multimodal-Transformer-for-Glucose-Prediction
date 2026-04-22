# Supervised vs Non-Invasive Glucose Modeling — A 33-Phase Comparison

This document gives a step-by-step comparison between the two projects in this repository:

- `glucose_transformer/` — supervised multimodal glucose forecasting
- `noninvasive_glucose/` — biosignal-only current-glucose estimation

The point is not to declare one project “better.” The point is to make explicit how the problem definition changes the architecture, training loop, calibration workflow, and interpretation of results.

## Problem Framing

### 1. Target definition

**Supervised:** predict future glucose at `t+30` and `t+60` minutes.  
**Non-invasive:** estimate glucose at the current time `t`.

### 2. Inference-time glucose availability

**Supervised:** recent glucose context is available and explicitly used as input.  
**Non-invasive:** glucose is never an inference-time input.

### 3. Primary research question

**Supervised:** can a multimodal Transformer model glucose dynamics over time?  
**Non-invasive:** can biosignals alone recover enough metabolic state to estimate current glucose?

### 4. Deployment interpretation

**Supervised:** CGM-assisted or CGM-conditioned forecasting.  
**Non-invasive:** sensor-free glucose estimation with only sparse onboarding labels.

### 5. Dominant source of predictive power

**Supervised:** recent glucose trajectory plus physiology.  
**Non-invasive:** indirect physiological correlates only.

### 6. Consequence for uncertainty

**Supervised:** uncertainty is useful but not central to the base architecture.  
**Non-invasive:** uncertainty is a first-class output because the mapping is latent and ambiguous.

## Data and Supervision

### 7. Core dataset

**Supervised:** OhioT1DM provides the clinical forecasting target and base HR signal.  
**Non-invasive:** synthetic multimodal cohort is the default fallback, with PhysioCGM as the intended real-data target.

### 8. Label role

**Supervised:** glucose is both a label and part of the input context.  
**Non-invasive:** glucose is only a label during training and sparse calibration.

### 9. Window length

**Supervised:** 2-hour history (`24` five-minute steps).  
**Non-invasive:** 30-minute history (`6` five-minute steps).

### 10. Why the window changes

**Supervised:** forecasting needs longer temporal context to model delayed effects.  
**Non-invasive:** current-state estimation needs recent autonomic and activity evidence more than stale history.

### 11. Modalities

**Supervised:** HR, glucose context, ECG-HRV, EMG, EEG, CBF.  
**Non-invasive:** HR, ECG-HRV, EMG, EEG band powers, CBF.

### 12. Synthetic data role

**Supervised:** synthetic ECG/EMG and later EEG/CBF scaffold the fusion experiments on top of OhioT1DM.  
**Non-invasive:** the full end-to-end pipeline can run on synthetic multimodal users even without an external real dataset adapter.

## Model Architecture

### 13. Foundational encoder

**Supervised:** starts with `TemporalTransformer`, a sequence encoder with a learnable CLS token.  
**Non-invasive:** starts directly with multiple modality encoders because glucose context is absent and fusion is required from the beginning.

### 14. Primary sequence anchor

**Supervised:** HR plus glucose context is the primary sequence.  
**Non-invasive:** HR alone is the anchor sequence, with other modalities contributing context.

### 15. Fusion design

**Supervised:** cross-attention is introduced gradually in Part B after Parts A and early baselines.  
**Non-invasive:** HR-centric cross-attention is built in as the base fusion mechanism.

### 16. Output head

**Supervised:** deterministic regression head predicts two future values.  
**Non-invasive:** `UncertaintyHead` predicts a mean and a log-variance.

### 17. Sequence summary token

**Supervised:** CLS token summarises the full sequence for forecasting.  
**Non-invasive:** CLS token summarises the fused multimodal sequence for current-glucose estimation.

### 18. User conditioning

**Supervised:** added in Part D through user embeddings and archetype priors.  
**Non-invasive:** preserved as a core deployment mechanism because sparse calibration is essential.

### 19. EEG handling

**Supervised:** explicitly benchmarked with frequency, patch, and hierarchical variants.  
**Non-invasive:** uses the cheapest and most interpretable form, EEG band powers.

### 20. Architectural goal

**Supervised:** maximise temporal predictive structure under controlled multimodal extension.  
**Non-invasive:** maximise physiological inference per watt and per label.

## Training and Optimisation

### 21. Base loss

**Supervised:** mean squared error on normalised future glucose targets.  
**Non-invasive:** Gaussian negative log-likelihood on current glucose.

### 22. Why the loss differs

**Supervised:** the model is partially observing the latent state through glucose context, so a point estimate is reasonable.  
**Non-invasive:** the latent state is not directly observed, so variance prediction is necessary.

### 23. Warmup and optimisation

**Supervised:** inverse-square-root warmup stabilises attention training from Part A onward.  
**Non-invasive:** reuses the same style of careful Transformer optimisation.

### 24. Encoder pretraining

**Supervised:** learning is mostly end-to-end within the staged parts.  
**Non-invasive:** modality-specific pretraining is more important because biosignal-only supervision is weaker.

### 25. Mixed precision pressure

**Supervised:** becomes critical in multimodal Parts B and C.  
**Non-invasive:** still useful, but the smaller 6-token window reduces pressure relative to Part C.

### 26. Calibration objective

**Supervised:** meta-learning is evaluated as a research question about adaptation.  
**Non-invasive:** calibration is a deployment requirement, not just an experiment.

## Personalisation and Adaptation

### 27. Few-shot learning role

**Supervised:** Part D asks how quickly the model adapts to new synthetic users.  
**Non-invasive:** adaptation is the mechanism by which the model becomes clinically usable for a new wearer.

### 28. What gets updated

**Supervised:** FOMAML adapts the user-conditioned backbone in the meta-learning framework.  
**Non-invasive:** only the user embedding is updated during calibration.

### 29. Calibration labels

**Supervised:** support/query splits are part of the meta-learning experiment design.  
**Non-invasive:** the calibration support set is intentionally chosen to span fasting, post-meal, and post-exercise states.

### 30. Consequence of over-updating

**Supervised:** aggressive adaptation can overfit a task.  
**Non-invasive:** updating the full network would overfit immediately to three labels and destroy the shared physiology prior.

## Evaluation and Trust

### 31. Primary metrics

**Supervised:** RMSE, MAE, Clarke Error Grid, ablations, benchmarks, adaptation curves, interpretability outputs.  
**Non-invasive:** RMSE, MAE, Clarke Error Grid, coverage, sharpness, reliability, calibration improvement, attribution.

### 32. Trust question

**Supervised:** “Is the model using modalities and lags in a biologically coherent way?”  
**Non-invasive:** “Can the model know when biosignals are insufficient, and can it safely improve after sparse calibration?”

### 33. Final tradeoff

**Supervised:** lower error is easier because glucose context is available at inference time.  
**Non-invasive:** higher error is acceptable if the system eliminates continuous invasive sensing while remaining calibrated, interpretable, and quickly personalisable.

## Bottom Line

The supervised project teaches the model family how to represent temporal physiology when glucose context is available. The non-invasive project removes that crutch and replaces it with probabilistic prediction, tighter reliance on physiology, and deployment-time calibration. They are not duplicate codebases. They are two answers to two different clinical questions:

- **Where is glucose going next, given recent glucose and physiology?**
- **What is glucose right now, given only physiology?**

Understanding that distinction is the key to understanding why both projects belong in the same repository.

