# Documentation Index

This folder consolidates the theory notes, technical runbooks, quantitative results, and interview-facing summaries for both projects in this repository.

## Start Here

If you are new to the repository, read these in order:

1. [Supervised overview](theory/01_supervised_overview.md)
2. [Non-invasive overview](theory/07_noninvasive_overview.md)
3. [Architecture guide](technical/architecture.md)
4. [Training guide](technical/training_guide.md)
5. [Comparison table](results/comparison_table.md)

## Folder Guide

### `theory/`

- [01_supervised_overview.md](theory/01_supervised_overview.md) — Bird's-eye view of the supervised five-part project
- [02_part_a_foundations.md](theory/02_part_a_foundations.md) — Temporal attention foundations from Part A
- [03_part_b_fusion.md](theory/03_part_b_fusion.md) — Multimodal fusion and cross-attention
- [04_part_c_efficient.md](theory/04_part_c_efficient.md) — Efficient EEG handling and long-sequence constraints
- [05_part_d_generalisation.md](theory/05_part_d_generalisation.md) — User embeddings and meta-learning
- [06_part_e_interpretability.md](theory/06_part_e_interpretability.md) — Interpretability and trust analyses
- [07_noninvasive_overview.md](theory/07_noninvasive_overview.md) — Biosignal-only glucose estimation rationale
- [08_comparison.md](theory/08_comparison.md) — Step-by-step supervised vs non-invasive comparison

### `technical/`

- [architecture.md](technical/architecture.md) — Shapes, modules, and signal flow
- [training_guide.md](technical/training_guide.md) — How to run each project
- [evaluation_guide.md](technical/evaluation_guide.md) — How metrics and saved artifacts are produced
- [hardware_constraints.md](technical/hardware_constraints.md) — 6GB VRAM decision log
- [dataset_guide.md](technical/dataset_guide.md) — Dataset access, placement, and licensing

### `results/`

- [supervised_results.md](results/supervised_results.md) — Status of committed supervised artifacts
- [noninvasive_results.md](results/noninvasive_results.md) — Current non-invasive smoke metrics
- [comparison_table.md](results/comparison_table.md) — Honest side-by-side comparison across the two projects

### `interview/`

- [preparation_guide.md](interview/preparation_guide.md) — Concise talking points, likely questions, and caveats

### `figures/`

- Curated repository-level figures that are small enough to commit and stable enough to reference from documentation.

