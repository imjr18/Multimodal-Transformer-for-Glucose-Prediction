# Supervised Glucose Forecasting — Parts A through E

Multimodal Transformer models for predicting blood glucose 30 and 60 minutes ahead from heart rate, ECG-derived HRV, EMG, EEG, and cerebral blood flow.

## Architecture Summary

The supervised project starts with `TemporalTransformer` in Part A, a compact pre-layer-normalised Transformer encoder with `d_model=64`, `n_heads=4`, `n_encoder_layers=2`, a learnable CLS token, and a two-output regression head. Part B replaces single-stream modeling with three multimodal fusion strategies and makes `CrossModalTransformer` the primary comparison point. Part C extends that backbone into `FullModalTransformer`, which adds EEG and CBF while constraining memory through efficient EEG encoders. Part D wraps the Part C backbone with user conditioning and first-order MAML. Part E keeps the trained model fixed and adds interpretability and trust analyses.

## Parts

### Part A — Foundations
**What it builds:** Vanilla Transformer encoder, HR-conditioned glucose forecasting  
**Key concept:** Self-attention, positional encoding, CLS token  
**Entry point:** `python glucose_transformer/run_part_a.py`  
**Theory:** [Part A foundations](../docs/theory/02_part_a_foundations.md)

### Part B — Multimodal Fusion
**What it builds:** Cross-modal attention with ECG-HRV and EMG  
**Key concept:** Cross-attention, modality type embeddings, ablation study  
**Entry point:** `python glucose_transformer/run_part_b.py`  
**Theory:** [Part B fusion](../docs/theory/03_part_b_fusion.md)

### Part C — Efficient Attention
**What it builds:** Frequency, patch, and hierarchical EEG encoders, plus CBF integration  
**Key concept:** Long-sequence attention constraints, patch tokenisation, gradient accumulation  
**Entry point:** `python glucose_transformer/run_part_c.py`  
**Theory:** [Part C efficient attention](../docs/theory/04_part_c_efficient.md)

### Part D — Population Generalisation
**What it builds:** User embeddings, archetype priors, FOMAML on a synthetic cohort  
**Key concept:** Distribution shift, low-shot personalisation, adaptation curves  
**Entry point:** `python glucose_transformer/run_part_d.py`  
**Theory:** [Part D generalisation](../docs/theory/05_part_d_generalisation.md)

### Part E — Interpretability
**What it builds:** Attention rollout, integrated gradients, probing classifiers, spurious-noise control, head analysis  
**Key concept:** Biological validity, trust, and internal representation analysis  
**Entry point:** `python glucose_transformer/run_part_e.py`  
**Theory:** [Part E interpretability](../docs/theory/06_part_e_interpretability.md)

## File Structure

```text
glucose_transformer/
├── preprocessing/          # Data ingestion, feature generation, simulation helpers
├── part_a/                 # Vanilla temporal Transformer + LSTM baseline
├── part_b/                 # Early / late / cross-attention multimodal fusion
├── part_c/                 # EEG efficiency experiments + full multimodal backbone
├── part_d/                 # Population generalisation, synthetic cohort, FOMAML
├── part_e/                 # Interpretability and trust analyses
├── data/
│   └── README_DATA.md      # OhioT1DM download and placement instructions
├── run_part_a.py
├── run_part_b.py
├── run_part_c.py
├── run_part_d.py
└── run_part_e.py
```

## Results

This repository snapshot does not preserve the final result CSV/JSON artifacts for Parts A through E. The training and evaluation code is present, and the expected output paths are wired in each part config, but files such as `part_b/results/fusion_comparison.csv`, `part_c/results/benchmark.csv`, and `part_d/results/cross_archetype_results.csv` are not committed here.

See [docs/results/supervised_results.md](../docs/results/supervised_results.md) for the exact artifact status per part and the quantitative values that *are* available from the committed code snapshot, such as model sizes, configured batch sizes, and the operational default that carries the `frequency_eeg` backbone into Part D.

## Key Findings

- The codebase fully implements the five-stage supervised learning path, but the committed repository is missing the final benchmark and evaluation artifacts needed to support publication-quality numeric claims.
- Part D carries `frequency_eeg` forward as the default EEG backbone, which strongly suggests the simpler representation was operationally preferred even though the benchmark CSV is absent.
- The interpretability stack in Part E is broad and well integrated, but its empirical outputs are not committed here, so no Part E biological claim is presented as established evidence.
- The most important reproducibility gap is artifact preservation, not missing model code: the repo can rerun the experiments, but it does not yet ship the resulting tables and figures.

