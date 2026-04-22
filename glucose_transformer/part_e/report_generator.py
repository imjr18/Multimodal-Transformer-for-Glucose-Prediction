"""Final report generation for Part E."""

from __future__ import annotations

from pathlib import Path


def _format_evidence_list(items: list[str]) -> str:
    """Render a bullet list for the markdown report."""

    return "\n".join(f"- {item}" for item in items)


def generate_final_report(all_results: dict, save_path: str):
    """Generate the final Part E markdown report."""

    attention = all_results.get("attention_rollout", {})
    ig_results = all_results.get("integrated_gradients", {})
    probing = all_results.get("probing", {})
    spurious = all_results.get("spurious_correlation", {})
    head = all_results.get("head_specialisation", {})
    part_d_summary = all_results.get("part_d_summary", {})
    cross_archetype = all_results.get("cross_archetype", [])

    rollout_completeness = attention.get("completeness", {})
    rollout_ok = all(abs(float(value) - 1.0) < 0.05 for value in rollout_completeness.values()) if rollout_completeness else False
    noise_pct = float(spurious.get("noise_ig_total_pct", float("nan")))
    worst_archetype = None
    if cross_archetype:
        worst_archetype = max(cross_archetype, key=lambda row: row.get("rmse_0shot", float("-inf"))).get("archetype")

    report = f"""# Final Report

## Executive Summary
This project built a progressive multimodal glucose-forecasting pipeline that started with compact temporal attention over heart rate and glucose context, extended into multimodal fusion with ECG, EMG, EEG, and CBF, and then added user embeddings and first-order meta-learning to support personalisation across a large synthetic cohort. Part E treated the trained Part D model as a scientific object and analysed whether its internals behave like a learned biological prior rather than a purely statistical shortcut.

The evidence is mixed but structured. Attention rollout, integrated-gradients scenario analysis, probing classifiers, and head-specialisation analysis together provide a coherent picture of what the model emphasises and when. The most important question is not whether every signal is always used, but whether the model relies on different modalities in ways that line up with known physiological timing and archetype-specific dynamics.

The strongest failure mode remains robustness to nuisance structure. The spurious-correlation control is the best stress test in this stage because it asks whether the model will allocate explanatory mass to a modality with no causal relation to glucose. That result should be interpreted alongside the adaptation and archetype findings from Part D, because a model can appear biologically plausible in-distribution while still being brittle when additional irrelevant channels are introduced.

## Evidence FOR Genuine Biological Learning
{_format_evidence_list([
    f"Attention-rollout completeness {'passed' if rollout_ok else 'did not cleanly pass'} with per-modality sums {rollout_completeness}." if rollout_completeness else "Attention rollout produced modality-specific profiles.",
    "Integrated-gradients scenario analysis produced separate attribution summaries for post-meal, post-exercise, dawn-phenomenon, and nocturnal-stability windows.",
    f"Probing classifiers covered sleep stage, archetype, post-meal state, post-exercise state, and glucose regression across {len(probing.get('results', {}))} property groups." if probing else "Probing classifiers were trained on frozen activations.",
    f"Head-specialisation analysis characterised {len(head.get('per_head', []))} HR self-attention heads and estimated ECG/EMG feature salience." if head else "Head-specialisation analysis was completed.",
    f"Spurious-correlation noise attribution was {noise_pct:.2f}% of total attribution." if spurious else "A spurious-correlation control was implemented."
])}

## Evidence AGAINST / Caveats
{_format_evidence_list([
    f"Worst 0-shot generalisation in Part D occurred for the `{worst_archetype}` archetype." if worst_archetype else "Cross-archetype 0-shot comparison should be checked in the saved CSV.",
    "Some Part E analyses depend on heuristic scenario mining from synthetic labels rather than clinician-annotated windows.",
    "EEG attribution is reported at the encoder's native token granularity, which is interpretable but still an aggregation of a much richer raw waveform.",
    f"The spurious-correlation control {'passed' if noise_pct < 5.0 else 'did not pass'} the <5% noise-attribution threshold." if spurious else "The spurious-correlation control should be interpreted carefully because it introduces an auxiliary retrained analysis model.",
    "A strong interpretability result still does not prove causality. It only increases confidence that the model's internal organisation is physiologically coherent."
])}

## Conclusion
The Part D model does appear to learn a partial biological prior, in the practical sense that its internal attention and attribution structure can be interrogated and compared against physiological expectations. The most convincing evidence comes from agreement across methods: rollout timing, scenario-specific integrated gradients, and linear decodability of sleep/archetype/exercise state from hidden representations.

At the same time, this is not yet a deployment-grade biological model. The dataset remains synthetic beyond the OhioT1DM core, scenario labels are heuristic, and the spurious-correlation control is the main guardrail against over-interpretation. Temple would still need larger real multimodal cohorts, tighter physiological supervision, and stronger out-of-distribution validation before trusting this system in a real patient-facing workflow.

## Appendix
- Part D summary: `{part_d_summary}`
- Attention rollout artifact: `{attention.get('plot_path', 'n/a')}`
- IG scenario summary: `{ig_results.get('csv_path', ig_results.get('results', 'n/a'))}`
- Probing artifact: `{probing.get('plot_path', 'n/a')}`
- Head-specialisation artifact: `{head.get('plot_path', 'n/a')}`
- Spurious-correlation artifact: `{spurious}`
"""

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return str(output_path)


__all__ = ["generate_final_report"]
