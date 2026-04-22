# Part C — Efficient Transformers for Long Biosignal Sequences

Part C is where the easy version of attention stops working. Heart rate, glucose, ECG-HRV, EMG, and CBF all remain cheap at the token counts used in Parts A and B. EEG does not. A 2-minute raw EEG segment at 256 Hz contains 30,720 samples, and naive self-attention over that sequence is mathematically incompatible with a 6GB GPU budget.

## 1. Why the problem is structural

The core issue is not PyTorch inefficiency. It is the quadratic attention matrix. For a sequence length `L`, self-attention materialises `L x L` pairwise interactions. At `L = 30,720`, a single float32 attention matrix already needs roughly 3.8 GB before gradients, activations, optimizer state, and model weights are even considered.

That is why Part C starts by explicitly demonstrating the failure. The efficient alternatives only make sense once the baseline infeasibility is concrete.

## 2. The three efficient strategies

Part C compares three ways of making EEG tractable:

- **Frequency features:** convert each 1-second EEG segment into relative band powers and attend over 120 interpretable tokens.
- **Patch encoding:** treat non-overlapping 64-sample EEG patches as tokens and attend over 480 local waveform summaries instead of 30,720 raw samples.
- **Hierarchical encoding:** summarise short 5-second windows locally, then attend globally over the sequence of local summaries.

Each strategy makes a different tradeoff between raw detail, structure, memory, and interpretability.

## 3. Why EEG and CBF matter biologically

EEG adds a window into sleep stage and arousal state. That matters for glucose because deep sleep, light sleep, wakefulness, cortisol dynamics, and autonomic balance all influence glucose regulation on the 30 to 60 minute horizon.

CBF is slower and lower dimensional, but it may carry the most direct brain-side context. In the Temple framing, reduced cerebral blood flow can perturb the hypothalamic systems that regulate glucose through sympathetic tone, hormones, and behavioural state.

## 4. What the full multimodal model should learn

The Part C backbone keeps HR-centric fusion from Part B and adds EEG and CBF as extra sources of context. HR, ECG, and EMG still provide the immediate peripheral physiological story. EEG provides neural-state context. CBF provides slow cerebrovascular context. The best-performing efficient EEG encoder becomes the new backbone for the later parts.

## 5. Why sleep-stage analysis is the scientific contribution

Efficient modeling is only half of Part C. The more interesting question is whether the model uses EEG in a physiologically meaningful way. If deep-sleep windows concentrate importance on delta-dominant segments and awake windows shift importance toward beta-dominant segments, the model is not just handling EEG efficiently. It is using EEG in the way the underlying physiology suggests it should.

