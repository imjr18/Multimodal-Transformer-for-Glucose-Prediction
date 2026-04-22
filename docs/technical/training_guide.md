# Training Guide

## Prerequisites

- Python `3.10+`
- PyTorch-compatible CUDA GPU with `6GB+` VRAM for the intended training path
- Roughly `10GB+` free disk space if you generate synthetic cohorts locally
- A virtual environment is strongly recommended

## Installation

From the repository root:

```bash
pip install -r requirements.txt
```

## Dataset Setup

See [dataset_guide.md](dataset_guide.md) for full details. The short version:

- OhioT1DM goes under `glucose_transformer/data/raw/`
- Real non-invasive data, if available, goes under `noninvasive_glucose/data/raw/`
- The non-invasive pipeline can run end to end on its synthetic fallback without an external raw dataset

## Running the Supervised Project

### Part A

```bash
python glucose_transformer/run_part_a.py
```

What it does:

- preprocesses OhioT1DM if processed windows are missing
- trains `TemporalTransformer`
- optionally trains the LSTM baseline
- evaluates on the test split
- writes attention visualisations if the model and data are available

Expected outputs:

- checkpoints under `glucose_transformer/part_a/checkpoints/`
- processed windows under `glucose_transformer/data/processed/`
- terminal summary with Transformer and LSTM RMSE if the run completes

Current repo note:

- the committed snapshot does **not** include the final Part A metric artifact, so reproduce the run locally if you need the exact test RMSE table

### Part B

```bash
python glucose_transformer/run_part_b.py
```

What it does:

- builds synthetic ECG-HRV and EMG features
- trains early, late, and cross-attention fusion models
- runs fusion comparison and modality ablation

Expected outputs:

- `glucose_transformer/part_b/results/fusion_comparison.csv`
- `glucose_transformer/part_b/results/ablation_results.csv`
- cross-attention heatmap and modality contribution figure

Current repo note:

- those Part B result files are not committed in this snapshot

### Part C

```bash
python glucose_transformer/run_part_c.py
```

What it does:

- demonstrates why vanilla EEG attention is infeasible
- builds EEG and CBF augmentations
- trains full multimodal models with different EEG encoders
- benchmarks RMSE, VRAM, and latency

Expected outputs:

- `glucose_transformer/part_c/results/benchmark.csv`
- `glucose_transformer/part_c/figures/sleep_stage_attention.png`

Current repo note:

- the benchmark CSV is not committed in this snapshot

### Part D

```bash
python glucose_transformer/run_part_d.py
```

What it does:

- generates the 1,000-user synthetic cohort
- meta-trains the user-conditioned Part C backbone with FOMAML
- produces adaptation and cross-archetype summaries

Expected outputs:

- synthetic cohort files under `glucose_transformer/data/synthetic_cohort/`
- `glucose_transformer/part_d/results/cross_archetype_results.csv`
- `glucose_transformer/part_d/figures/adaptation_curve.png`
- `glucose_transformer/part_d/figures/embedding_space.png`

Current repo note:

- these Part D artifacts are not committed in this snapshot

### Part E

```bash
python glucose_transformer/run_part_e.py
```

What it does:

- runs attention rollout
- computes integrated gradients for biological scenarios
- trains probing classifiers
- performs the spurious-correlation control
- analyses head specialisation
- writes the final markdown report

Expected outputs:

- `glucose_transformer/part_e/results/ig_scenario_summary.csv`
- `glucose_transformer/part_e/results/spurious_correlation.json`
- `glucose_transformer/part_e/FINAL_REPORT.md`

Current repo note:

- the committed snapshot contains the code, not the final Part E artifacts

## Running the Non-Invasive Project

### End-to-end pipeline

```bash
python run_noninvasive.py
```

Useful flags:

- `--simulate_only`
- `--pretrain_only`
- `--eval_only`
- `--no_cuda`
- `--force_rebuild`

What it does:

- builds or loads the processed synthetic cohort windows
- optionally pretrains the EEG and ECG encoders
- fine-tunes the full non-invasive model
- runs uncertainty evaluation, calibration demo, baseline comparison, and attribution

Current committed smoke-run outputs:

- baseline RMSE: `21.81 mg/dL`
- baseline MAE: `20.99 mg/dL`
- Zone A+B: `100.0%`
- coverage at nominal 95%: `100.0%`
- mean 95% interval width: `125.35 mg/dL`

These numbers come from the committed smoke artifacts under `noninvasive_glucose/results/` and should not be treated as final convergence metrics.

## Practical Run Order

If you want to reproduce the repository from scratch:

1. Install dependencies.
2. Set up OhioT1DM if you want the supervised pipeline.
3. Run Part A, then Part B, then Part C, then Part D, then Part E.
4. Run `python run_noninvasive.py` after the supervised project is understood.
5. Copy the resulting small CSV/JSON summaries into documentation if you plan to publish updated metrics.

