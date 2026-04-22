# Interview Preparation Guide

This guide is written for someone presenting this repository as a serious ML systems project.

## 60-second project pitch

I built two related glucose modeling systems around compact Transformers under a strict 6GB VRAM budget. The first is a staged supervised forecasting project that starts with a heart-rate-conditioned Transformer and grows into a multimodal backbone over HR, ECG-derived HRV, EMG, EEG, and cerebral blood flow, then adds user conditioning, FOMAML, and interpretability. The second is a standalone non-invasive estimator that removes glucose from the inference path entirely, predicts current glucose from biosignals only, outputs both a mean and uncertainty, and calibrates to a new user with three sparse reference readings by updating only a 16-dimensional user embedding.

The supervised repository is code-complete but does not currently preserve the final result artifacts in its committed snapshot, so I describe its architecture and evaluation design precisely and avoid inventing numbers. The non-invasive repository does preserve smoke-run outputs: `21.81 mg/dL` RMSE against a `26.48 mg/dL` population-mean baseline, `100%` nominal coverage with overly wide intervals, and a small calibration demo that improved a held-out slice from `22.16` to `18.55 mg/dL`.

## Questions you are likely to get

### Explain self-attention in your implementation

In Part A, each timestep is embedded into a `64`-dimensional token and split across `4` heads, so each head operates on `16` dimensions. Self-attention computes query, key, and value projections for each token and builds a `25 x 25` attention pattern once the CLS token is included. That lets the model compare every point in the 2-hour window directly instead of carrying old information through a recurrent hidden state.

### Why did you use cross-attention for multimodal fusion?

Because multimodal fusion is not just feature concatenation. HR, ECG-HRV, and EMG are different physiological views of the same event, often with different lags. Cross-attention lets the HR representation ask targeted questions of ECG, EMG, EEG, and CBF instead of mixing everything before the model understands what the modalities mean.

### How did you handle EEG under a 6GB VRAM limit?

I explicitly avoided raw self-attention over the full 2-minute EEG sequence because `30,720` tokens would explode quadratically. I implemented three alternatives: `120` one-second frequency tokens, `480` patch tokens of `64` samples each, and a hierarchical encoder over `24` local windows. The full-model parameter counts stayed around `610k-627k`, which kept the system manageable.

### Why does the non-invasive model use NLL instead of MSE?

Because the biosignal-to-glucose mapping is ambiguous and heteroscedastic. The model should be able to say “this window is uncertain” instead of forcing every case into a single hard point estimate. The `UncertaintyHead` outputs a mean and log-variance, and the Gaussian NLL penalises both being wrong and being dishonestly uncertain.

### What did calibration do in practice?

In the committed smoke artifact, the calibration demo improved a held-out user slice from `22.16` to `18.55 mg/dL`, about a `16.28%` relative improvement. The important design choice is that calibration updates only the `16`-dimensional user embedding, not the encoders, so the shared physiology prior stays intact.

### What did the non-invasive uncertainty results show?

They showed that the uncertainty plumbing works, but the current weights are undertrained. The committed artifact has `100%` nominal 95% coverage, which sounds good until you see the mean interval width is about `125.35 mg/dL`. That means the intervals are far too wide and the model is underconfident.

## Questions to answer carefully

### Is the supervised project publication-ready?

Architecturally yes, artifact-wise not yet. The repository contains the full supervised implementation, but the committed snapshot does not preserve the final Part A-E result CSV/JSON files. I would present it as code-complete and rerunnable, but I would not claim final supervised RMSE numbers without regenerating those artifacts.

### Is the non-invasive model clinically useful today?

No. The committed non-invasive metrics are smoke metrics, not convergence metrics. The model beats a population-mean baseline, but its uncertainty is too wide and the result bundle is still too small to support clinical claims.

### What is the main technical weakness of the current repository?

Artifact preservation. The hardest gap is not missing code; it is missing committed summaries from the supervised runs. Without those, architecture review is easy but empirical review is incomplete.

## Good closing line

The strongest thing about this repository is that it makes the design tradeoffs inspectable: why the model stays small, why EEG had to be compressed, why user adaptation lives in a compact embedding, and why uncertainty is central once glucose leaves the inference path.

