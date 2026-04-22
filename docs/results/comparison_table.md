# Supervised vs Non-Invasive Comparison Table

This table compares what is actually committed for the two projects in the current repository snapshot.

| Dimension | Supervised project | Non-invasive project |
|-----------|--------------------|----------------------|
| Prediction task | Forecast glucose at `+30` and `+60` min | Estimate current glucose |
| Inference-time glucose input | Yes | No |
| Default supervised window | `24` steps = `2` hours | `6` steps = `30` minutes |
| Output head | Deterministic regression head | `UncertaintyHead` with mean + log-variance |
| Core uncertainty mechanism | Not part of base forecasting head | Gaussian NLL + MC Dropout |
| Personalisation | Part D user embeddings + FOMAML | User embedding + deployment calibration |
| Calibrates at deployment | Not the main deployment target | Yes, from `3` sparse readings |
| Trainable parameter count (base full model) | `610,178` for Part C `frequency_eeg` backbone | `627,394` |
| Committed final RMSE artifact | Not preserved in snapshot | `21.81 mg/dL` smoke artifact |
| Committed Zone A+B artifact | Not preserved in snapshot | `100.0%` smoke artifact |
| Committed calibration improvement artifact | Not preserved in snapshot | `22.16 -> 18.55 mg/dL` on smoke demo |
| Committed uncertainty artifact | Not preserved in snapshot | `100%` coverage, `125.35 mg/dL` sharpness |
| Artifact completeness | code-complete, artifact-light | code-complete, smoke-artifact present |

## Practical Interpretation

- The supervised codebase is further along conceptually, but the current repository snapshot does not preserve the numeric outputs needed for a final comparison table.
- The non-invasive codebase does preserve a small result bundle, but those numbers come from a smoke run and should be read as execution evidence rather than final performance.
- The exact measured accuracy gap between the two projects is therefore **not yet recoverable from the committed repo snapshot**.

## What to do before publication

1. Rerun the supervised stages and preserve their small summary CSV/JSON outputs.
2. Rerun the non-invasive pipeline to convergence and keep the resulting summary tables.
3. Regenerate this comparison table with paired, committed artifacts from the same repository state.

