# Part A — Foundations of Temporal Attention

This document mirrors the reasoning that drives the Part A implementation. The goal is not just to have working code. The goal is to understand why this particular small Transformer is structured the way it is and what each piece contributes to single-signal glucose forecasting.

## 1. Why start with a Transformer instead of only an LSTM

An LSTM processes a sequence one step at a time. That makes intuitive sense for time-series data, but it also means information from older timesteps must be carried through every intermediate hidden-state update. For glucose forecasting, that is a real problem because the physiological relationship between a change in heart rate and a later glucose response can be delayed by roughly 30 to 45 minutes. A model that struggles with long-range credit assignment will often overweight the most recent few readings and underweight the more informative lag.

The Transformer removes that sequential bottleneck. Self-attention lets every timestep compare itself with every other timestep directly. With a 24-step input window, the model can inspect the full 2-hour history in one pass and learn that events from 6 to 9 steps earlier may matter more than the latest point.

## 2. Why scalar biosignals need token embeddings

Heart rate and glucose arrive as scalar values. Self-attention expects vectors. The embedding layers therefore project each scalar into a `d_model=64` feature space. This projection is not just a shape fix. It gives the model room to learn useful latent structure such as:

- whether heart rate is elevated versus baseline
- whether glucose is rising or falling
- whether the current reading is unusual relative to surrounding context

Part A keeps the embedding simple: one linear layer per signal. That is enough to create learnable token representations without adding unnecessary complexity.

## 3. Why glucose context is included with heart rate

The prompt focuses on heart rate as the primary input signal, but the model also receives historical glucose as context. That is deliberate. Forecasting glucose without knowing where glucose currently is throws away the strongest prior in the problem. The model should be able to reason jointly about:

- recent heart-rate dynamics
- the recent glucose trajectory
- how those two histories interact before the forecast horizon

The implementation embeds heart rate and glucose separately, concatenates those embeddings, and projects them back down to `d_model`. This keeps the two signals distinct long enough to learn different signal-specific representations, then forces the encoder to work with a fused temporal token stream.

## 4. Why positional encoding is mandatory

Self-attention is permutation-equivariant. If you shuffle the token order and do not inject positional information, the model has no way to know which timestep came first. That is unacceptable for physiological time-series.

Sinusoidal positional encoding solves this by adding a deterministic, unique vector to every token position. The encoding is fixed rather than learned. That choice is useful here because it is lightweight, stable, and easy to interpret. Every token therefore carries two kinds of information:

- what happened at that timestep
- where that timestep sits inside the 2-hour history

## 5. Why self-attention is the core mechanism

For each token, self-attention learns three projections:

- Query: what this position is looking for
- Key: what this position offers
- Value: what content should be passed along if selected

The attention score between two positions is the dot product of a query and key. Softmax turns those scores into weights, and the output becomes a weighted sum of value vectors. In practice, this lets the model learn patterns such as:

- a heart-rate increase 35 minutes ago matters strongly for the forecast
- stable baseline readings are less relevant for the current prediction
- glucose context may dominate short-horizon prediction in some windows

Because the Part A sequence length is only 24, the quadratic cost of attention is trivial. This is an ideal setting for learning the mechanism cleanly.

## 6. Why multi-head attention matters

One attention mechanism can learn one notion of relevance. Multi-head attention lets the model learn several notions in parallel. With `n_heads=4` and `d_model=64`, each head operates on 16-dimensional subspaces. Different heads can specialise in different temporal questions, for example:

- one head may focus on the 30 to 45 minute lag band
- one head may focus on very recent glucose context
- one head may track longer, smoother temporal trends
- one head may compare local fluctuations to the full-window baseline

This specialisation is not hard-coded. It is an emergent property of training.

## 7. Why the encoder uses Pre-LN and a small feed-forward network

Each encoder layer contains:

1. multi-head self-attention
2. a position-wise feed-forward network

Residual connections make it easier to optimise because each block only needs to learn a useful update to its input. LayerNorm before the sublayers (`norm_first` in PyTorch) is the Pre-LN variant, which is usually more stable for smaller models and limited-data settings than the original Post-LN arrangement.

The feed-forward dimension is set to `256`, which is exactly `4 x d_model`. That is the standard Transformer expansion ratio and gives the model enough nonlinear capacity without pushing memory usage outside the 6GB GPU budget.

## 8. Why a CLS token is used instead of average pooling

The encoder outputs one vector per timestep. The forecasting head needs a single sequence summary. A learnable CLS token is prepended so the model can gather that summary through attention. This is preferable to mean pooling because it lets the network learn how to aggregate rather than forcing a fixed averaging rule.

If the task depends mainly on the 35-minute lag region, the CLS token can attend there more strongly. If the current glucose trajectory dominates the short-term forecast, it can shift attention toward recent context. The aggregation strategy is learned from data.

## 9. Why the regression head predicts both horizons jointly

The output head predicts glucose at `+30` and `+60` minutes with one shared encoder. This is a multitask setup. The two horizons are different, but they are not independent. They arise from the same physiology and the same 2-hour history. Sharing the encoder helps the model learn reusable temporal features while the final linear layer separates the two forecast targets.

## 10. Why the training recipe is careful

Transformers are sensitive to optimisation. Part A therefore uses:

- Adam with `betas=(0.9, 0.98)`
- weight decay for mild regularisation
- gradient clipping at `1.0`
- inverse-square-root warmup scheduling

The warmup schedule is important because attention weights are effectively random at the start of training. Large initial updates can destabilise the model before it has learned meaningful token interactions. Warmup lets the optimiser ramp up gradually and then decay smoothly.

## 11. Why RMSE alone is not sufficient

RMSE and MAE tell you how far predictions are from ground truth on average, but they do not tell you whether the mistakes are clinically tolerable. The Clarke Error Grid is therefore part of the evaluation stack. It categorises predictions by treatment risk:

- Zone A: clinically accurate
- Zone B: benign errors
- Zone C: overcorrection errors
- Zone D: dangerous failures to detect hypo or hyperglycaemia
- Zone E: erroneous treatment direction

For glucose forecasting, a model can have a decent RMSE and still make dangerous mistakes. That is why the pipeline reports both numeric error and Clarke zones.

## 12. What the attention plots should teach you

Attention visualisation is not a proof of causality, but it is still useful for sanity checking what the encoder has learned. In a healthy Part A result, you would expect to see some combination of:

- CLS attention spread across the full window when building a summary
- non-uniform heads that focus on informative lag regions
- some emphasis around the 30 to 45 minute band when the HR-glucose coupling is present

If every head is nearly uniform, the model is probably underfitting. If every head collapses onto only the most recent point, the model may be acting like a simple extrapolator instead of learning richer temporal structure.

## 13. Why Part A is intentionally constrained

Part A is not trying to be the final multimodal system. It deliberately avoids:

- multimodal fusion
- patient embeddings
- meta-learning
- integrated gradients
- federated or population-level adaptation

Those belong to later parts. Here the goal is to learn the foundation well: tokenisation, positional encoding, self-attention, CLS summarisation, careful optimisation, clinically meaningful evaluation, and basic attention inspection.

