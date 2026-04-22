# Multimodal Transformer for Non-Invasive Glucose Monitoring

> Predicting blood glucose from physiological biosignals using cross-modal attention Transformers — supervised forecasting and non-invasive estimation.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![GPU](https://img.shields.io/badge/GPU-6GB%20VRAM-red.svg)

## Overview

This repository packages two related research projects around glucose modeling from wearable physiology. The first project, `glucose_transformer/`, is a staged supervised forecasting system that predicts glucose 30 and 60 minutes ahead from heart rate, synthetic ECG-HRV, EMG, EEG, and cerebral blood flow. The second, `noninvasive_glucose/`, removes glucose entirely from inference-time inputs and estimates current glucose from biosignals alone, with explicit uncertainty and lightweight calibration.

The two-project structure is deliberate. The supervised system is the learning path: it introduces temporal self-attention, cross-modal fusion, efficient EEG handling, population generalisation, and interpretability one stage at a time. The non-invasive system then reuses those ideas under a harder deployment constraint: CGM acts only as a teacher during training and sparse onboarding, never as a live sensor at inference time.

The motivating application is a wearable system in the Temple-style sensing space: a compact multimodal model that can combine cardiac, neural, muscular, and cerebrovascular signals into a glucose-relevant latent state. This repository is therefore not only a set of models; it is also a reproducible documentation package for the design tradeoffs required to make that idea fit on a 6GB VRAM development target.

## Results at a Glance

| Model | RMSE (mg/dL) | Zone A+B | Signals | Notes |
|-------|-------------:|---------:|---------|-------|
| Part A: HR only | Artifact not committed | Artifact not committed | HR + glucose context | Evaluation code exists; final result JSON was not preserved in this snapshot |
| Part B: + ECG/EMG | Artifact not committed | Artifact not committed | HR, glucose context, ECG-HRV, EMG | `fusion_comparison.csv` and `ablation_results.csv` are not present in the committed tree |
| Part C: + EEG/CBF | Artifact not committed | Artifact not committed | All 5 supervised modalities | `benchmark.csv` was not preserved in the committed tree |
| Part D: Personalised | Artifact not committed | Artifact not committed | All 5 + user conditioning | Adaptation plots/results are implemented but not committed here |
| Non-Invasive | 21.81 | 100.0% | HR, ECG-HRV, EMG, EEG bands, CBF | Current committed artifact is a smoke run, not a full convergence report |

The supervised project is code-complete but artifact-light in this repository snapshot. See [docs/results/supervised_results.md](docs/results/supervised_results.md) for the exact status of each missing metric file, and [docs/results/noninvasive_results.md](docs/results/noninvasive_results.md) for the currently committed non-invasive smoke metrics.

## Architecture

```text
Supervised Backbone (Part C / Part D)

HR [B, 24, 1] --------------------┐
Glucose Ctx [B, 24, 1] ----------┐│
ECG-HRV [B, 24, 5] -------------┐││
EMG [B, 24, 2] ----------------┐│││
EEG --------------------------┐ ││││
  frequency:    [B, 120, 5]   │ ││││
  patch:        [B, 30720]    │ ││││
  hierarchical: [B, 30720]    │ ││││
CBF [B, 24, 1] ---------------┘ ││││
                                 ││││
Per-modality encoders -----------┘│││
  TokenEmbedding + PosEnc +        ││
  TransformerEncoder               ││
                                   ││
HR-centric cross-attention <-------┘│
  HR attends to ECG, EMG, CBF       │
  EEG contributes summary token ----┘
               │
               ▼
Final fusion encoder + CLS token
               │
               ▼
RegressionHead
Linear(64,64) -> ReLU -> Dropout -> Linear(64,2)
               │
               ▼
Forecasts: glucose_t+30, glucose_t+60

Non-Invasive Backbone

HR [B, 6, 1] + ECG [B, 6, 5] + EMG [B, 6, 2] + EEG bands [B, 6, 5] + CBF [B, 6, 1]
               │
               ▼
Per-modality encoders -> HR-centric cross-attention fusion -> user token
               │
               ▼
Final encoder + CLS
               │
               ▼
UncertaintyHead
Linear(64,64) -> ReLU -> Dropout -> Linear(64,2)
               │
               ├── mean glucose estimate
               └── log-variance
```

## Project Structure

```text
TempleTask/
├── README.md                     # Repository landing page
├── LICENSE                       # MIT license + dataset notices
├── .gitignore                    # Publication-specific ignore rules
├── requirements.txt              # Shared dependency set
├── docs/                         # Consolidated theory, technical docs, results
├── glucose_transformer/          # Supervised forecasting project
│   ├── run_part_a.py ... run_part_e.py
│   ├── preprocessing/
│   ├── part_a/ ... part_e/
│   └── data/README_DATA.md
└── noninvasive_glucose/          # Standalone non-invasive estimation project
    ├── run_noninvasive.py
    ├── simulation/
    ├── models/
    ├── training/
    ├── calibration/
    ├── evaluate/
    └── interpretability/
```

## Quick Start

### Supervised Glucose Forecasting

```bash
git clone https://github.com/your-username/glucose-transformer
cd glucose-transformer
pip install -r requirements.txt
# Download OhioT1DM (see docs/technical/dataset_guide.md)
python glucose_transformer/run_part_a.py
```

### Non-Invasive Estimation

```bash
# After reviewing the supervised project
pip install -r requirements.txt
# Download PhysioCGM or use the built-in synthetic fallback
python run_noninvasive.py
```

## Documentation

| Document | Description |
|----------|-------------|
| [Theory: Supervised](docs/theory/01_supervised_overview.md) | Bird's-eye view of the supervised five-part learning path |
| [Theory: Non-Invasive](docs/theory/07_noninvasive_overview.md) | Rationale for biosignal-only estimation and calibration |
| [Architecture Guide](docs/technical/architecture.md) | Full model architecture with tensor shapes |
| [Training Guide](docs/technical/training_guide.md) | How to reproduce runs end to end |
| [Interview Prep](docs/interview/preparation_guide.md) | Publication and interview-facing project summary |
| [Results](docs/results/supervised_results.md) | Committed artifact status and measured outputs |

## Learning Journey

This repository was built as a staged Transformer learning path:

| Part | Concept Learned | New Capability Added |
|------|-----------------|---------------------|
| A | Self-attention, positional encoding | HR-conditioned glucose forecasting |
| B | Cross-modal attention, multimodal fusion | + ECG-HRV and EMG |
| C | Efficient attention for long signals | + EEG and CBF |
| D | FOMAML, user embeddings | Population generalisation and adaptation |
| E | Integrated gradients, probing, rollout | Interpretability and trust analysis |

## Hardware Requirements

All experiments were designed around a 6GB VRAM GPU budget. Every major memory-related decision is documented in [docs/technical/hardware_constraints.md](docs/technical/hardware_constraints.md), including why `d_model=64`, why EEG is reduced or patched, and why the non-invasive system uses a 30-minute window.

## Datasets

| Dataset | Used For | Access |
|---------|----------|--------|
| OhioT1DM | Supervised forecasting and patient-level evaluation | See [dataset guide](docs/technical/dataset_guide.md) |
| PhysioCGM | Non-invasive estimation target dataset | See [dataset guide](docs/technical/dataset_guide.md) |
| MIT-BIH Arrhythmia | Optional ECG pretraining | [PhysioNet](https://physionet.org/content/mitdb/1.0.0/) |
| CHB-MIT Scalp EEG | Optional EEG pretraining | [PhysioNet](https://physionet.org/content/chbmit/1.0.0/) |

## Citation

If you use this code or find the repository structure useful, please cite:

```bibtex
@misc{multimodal_glucose_monitoring_2026,
  title={Multimodal Transformer for Non-Invasive Glucose Monitoring},
  author={Project Contributors},
  year={2026},
  url={https://github.com/your-username/glucose-transformer}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details. Dataset licenses and access requirements apply separately; see [docs/technical/dataset_guide.md](docs/technical/dataset_guide.md).

