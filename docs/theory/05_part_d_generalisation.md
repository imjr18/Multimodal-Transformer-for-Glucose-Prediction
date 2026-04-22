# Part D — Population Generalisation Across a Synthetic Cohort

Part D adds personalisation on top of the Part C multimodal backbone without rewriting the backbone itself. The core idea is to keep the shared biosignal encoder fixed in structure and condition it on compact user-specific context.

## Why user embeddings

Glucose forecasting is not only a signal-processing problem. Different users can have different resting heart rates, sleep patterns, exercise responses, baseline glucose levels, and cerebral blood-flow dynamics. A single population model averages over those differences. A user embedding gives the model a compact representation of who the current user is so the same backbone can behave slightly differently for different physiological profiles.

The embedding is injected into the CLS token before each encoder layer. This is cheap, because only one token is modified, but effective, because the CLS token is the global summary token that participates in attention with the entire sequence.

## Why archetype warm starts

New users do not have learned identity vectors yet. Starting them from a random embedding makes personalisation unstable and data-hungry. Starting from an archetype prototype gives the model a reasonable prior:

- `athlete`: low resting HR, strong exercise response
- `sedentary`: common baseline physiology
- `elderly`: lower CBF and more fragmented sleep
- `diabetic`: high glucose variability and lower insulin sensitivity

This makes zero-shot performance less brittle and gives the adaptation step a better initial point.

## Why first-order MAML

Second-order MAML differentiates through the inner-loop update itself. That is accurate but too memory-intensive for a Part C backbone that still contains the EEG pathway. First-order MAML keeps the same adaptation structure but drops the second-order term. In practice this captures most of MAML's benefit while remaining viable under the 6GB VRAM constraint.

## Why the cohort is stored as low-rate signals

A literal 7-day raw EEG trace at 256 Hz for 1,000 users would be prohibitively large. Part D therefore stores the tractable 5-minute physiological signals and regenerates the 2-minute EEG window on demand from the local HR and glucose context. This preserves the Part C modelling assumptions while keeping storage and I/O realistic.

