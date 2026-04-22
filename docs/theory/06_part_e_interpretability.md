# Part E — Interpretability, Trust, and Biological Validity

Part E does not build a new forecasting model. It treats the trained Part D model as a scientific object and asks whether the patterns it uses are biologically coherent or merely predictive shortcuts.

## Why multiple interpretability methods

No single interpretability method is sufficient here:

- Attention rollout shows how information can propagate through the model.
- Integrated gradients shows which input features most affect the output.
- Probing classifiers test what information is linearly decodable from hidden states.
- Head-specialisation analysis checks whether attention heads divide labour in a meaningful way.
- The spurious-correlation control asks whether the model will attach itself to irrelevant inputs if given the chance.

Together these analyses are much stronger than any one of them in isolation.

## Why CPU for integrated gradients

Integrated gradients requires many forward and backward passes. On a 6GB GPU that is unnecessary pressure for an analysis step. Running it in float32 on CPU is slower but simpler, deterministic, and memory-safe.

## Why checkpoint each analysis

Part E is the longest stage to rerun after interruption. Each major analysis saves its own outputs so a later run can resume from completed artifacts rather than recomputing rollout, scenario attribution, probing, or the spurious-noise control from scratch.

