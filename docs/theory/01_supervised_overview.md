# Supervised Glucose Forecasting — Full Overview

## What problem the supervised project solves

The supervised `glucose_transformer/` project predicts blood glucose 30 and 60 minutes into the future from physiological context. In its earliest stage, the model sees heart rate plus recent glucose context. By the end of the project, it becomes a multimodal, personalised, and interpretable forecasting system that can ingest heart rate, ECG-derived HRV, EMG, EEG, and cerebral blood flow while staying within a 6GB VRAM budget.

The design is educational as much as it is predictive. Each part adds one major idea to the same codebase rather than replacing the whole system. That means the repository is best understood as a sequence of controlled extensions:

1. Learn temporal self-attention on a small, stable problem.
2. Add multimodal fusion and compare fusion strategies.
3. Solve the long-sequence EEG problem without breaking the hardware budget.
4. Add population-level personalisation through user conditioning and meta-learning.
5. Interrogate the trained model with multiple interpretability methods.

## Why this was built in five parts

The five-part structure prevents the model from becoming opaque too early. If a single repository begins with cross-modal fusion, efficient EEG handling, meta-learning, and attribution all at once, it becomes difficult to isolate why any improvement or failure occurred. Here, each stage adds a single new pressure:

- Part A adds temporal attention.
- Part B adds modality interaction.
- Part C adds memory-constrained long-sequence modeling.
- Part D adds user variation and fast adaptation.
- Part E adds trust and interpretability.

Because the stages share the same underlying forecasting target, the later parts are cumulative rather than disconnected.

## Part A: foundations of temporal attention

Part A builds `TemporalTransformer`, a compact pre-layer-normalised Transformer encoder over 24 five-minute timesteps. It is intentionally small: `d_model=64`, `n_heads=4`, `n_encoder_layers=2`, and `d_ff=256`. The goal is not raw scale. The goal is to learn how self-attention behaves on a short physiological sequence where direct token-to-token comparison is cheap.

This stage teaches the essential Transformer mechanics:

- scalar biosignals must be embedded into a vector space
- positional encoding is mandatory because attention is order-agnostic by default
- a CLS token can act as a learned sequence summary
- warmup scheduling and gradient clipping matter for stable optimisation
- RMSE alone is not sufficient; clinical acceptability requires Clarke Error Grid analysis

It also establishes a baseline comparison against a 2-layer LSTM.

## Part B: multimodal fusion

Part B extends the Part A backbone into a multimodal system with HR, ECG-derived HRV, and EMG. OhioT1DM does not provide simultaneous ECG and EMG, so the repository generates physiologically coupled synthetic features from the observed HR and glucose context. This makes Part B an architecture experiment rather than a claim about newly collected multimodal data.

The important lesson in Part B is that multimodal modeling is not just “add more columns.” The repository compares three fusion strategies:

- early fusion
- late fusion
- HR-centric cross-attention

Cross-attention is the conceptually richest option because it lets one modality ask targeted questions of another rather than mixing everything before the model knows what the signals mean.

## Part C: efficient attention for EEG and CBF

Part C adds EEG and cerebral blood flow, which forces the repository to confront the quadratic cost of attention. A 2-minute EEG segment at 256 Hz contains 30,720 samples, which makes naïve attention infeasible on a 6GB GPU. The part therefore starts by demonstrating the failure, then implements three alternatives:

- frequency-band features
- patch-based EEG encoding
- hierarchical local-global encoding

This stage is important for two reasons. First, it turns a theoretical complexity issue into a concrete engineering constraint. Second, it tests whether the physiologically relevant EEG information for this task is low-frequency state information, such as sleep stage, rather than raw waveform detail. CBF is added as a slow contextual regulator rather than a fast event stream.

## Part D: population generalisation

Part D asks what happens when the model leaves the small supervised patient split and has to operate across a broader synthetic population. The answer in this repository is not to rewrite the backbone, but to condition it.

This stage adds:

- a 1,000-user synthetic cohort with four archetypes
- 16-dimensional user embeddings
- archetype prototype embeddings for warm starts
- first-order MAML for low-shot adaptation

The core idea is that the physiology-to-glucose relationship is shared, but baseline levels, exercise responses, sleep quality, and insulin sensitivity differ across users. User conditioning therefore becomes a lightweight way to steer a shared model toward a person-specific operating point.

## Part E: interpretability and trust

Part E keeps the trained Part D model fixed and treats it as an object for scientific analysis. Rather than adding predictive capacity, it adds instrumentation:

- attention rollout
- integrated gradients
- probing classifiers
- spurious-correlation control
- head specialisation analysis

The purpose is to answer a stronger question than “does it predict well?” The repository asks whether the model relies on modalities and time lags in ways that line up with known physiology, and whether it resists obviously irrelevant noise.

## How the five parts fit together

The coherent story across Parts A-E is:

- Attention learns temporal relevance.
- Cross-attention learns inter-modality relevance.
- Efficient attention keeps those ideas feasible on long signals.
- Meta-learning keeps them useful across users.
- Interpretability checks whether the learned structure is biologically plausible.

That sequence is why the supervised project is more than a model zoo. It is a progressive architecture curriculum built around one research question: can a multimodal Transformer learn a useful biological prior for glucose dynamics under realistic hardware constraints?

## What this supervised project is not

The supervised project is not a non-invasive deployment system. It still uses recent glucose context at inference time, especially in the early parts. That is a feature, not a flaw, because the repository is first teaching the model family how to reason about temporal physiology under supervision. The non-invasive project in this repository then takes those lessons and removes glucose from the inference path entirely.

## How to use the rest of the theory docs

Read the part-specific notes next:

- [Part A foundations](02_part_a_foundations.md)
- [Part B fusion](03_part_b_fusion.md)
- [Part C efficient attention](04_part_c_efficient.md)
- [Part D generalisation](05_part_d_generalisation.md)
- [Part E interpretability](06_part_e_interpretability.md)

Then read [the supervised vs non-invasive comparison](08_comparison.md) to see exactly how the second project departs from the first.

