# Part B — Cross-Modal Attention and Multimodal Fusion

Part B extends the Part A temporal Transformer from a single-signal setting to a multimodal one. The goal is not to throw more signals at the model blindly. The goal is to understand what changes when different physiological views of the same latent event are allowed to interact through attention.

## 1. Why multimodal fusion is different from “more features”

Heart rate, ECG-derived HRV, and EMG are not just three columns in a table. They are different projections of the same biology. A meal, stress response, or exercise event can alter all three, but not in identical ways and not always at the same temporal lag. A useful fusion model therefore has to answer two questions at once:

- what does each modality mean on its own
- when should one modality inform another

That is why Part B compares early fusion, late fusion, and cross-attention fusion instead of assuming one design is obviously correct.

## 2. Why synthetic ECG and EMG are acceptable here

OhioT1DM gives glucose and heart rate, but not simultaneous ECG and EMG. Rather than switching datasets and losing the clinical glucose forecasting setup, Part B generates synthetic ECG-HRV and EMG-envelope features that are explicitly tied to the observed heart-rate and glucose context.

This is a modelling exercise, not a claim of new physiology. The synthetic signals are a controlled scaffold that lets you test whether the fusion architecture can exploit cross-modal structure when that structure is present.

## 3. Why HRV is the right abstraction for ECG at this stage

Raw ECG is far too long for a standard Transformer under a 6GB VRAM budget, but that is only half the reason to avoid it. The glucose-relevant part of ECG in this setting is mostly the temporal spacing between beats, not the full waveform morphology. HRV features such as SDNN, RMSSD, LF power, HF power, and LF/HF ratio compress those autonomic patterns into a manageable representation.

That makes HRV a strong bridge modality between raw heart-rate dynamics and the slower glucose response.

## 4. What cross-attention adds beyond self-attention

In self-attention, one sequence interrogates itself. In cross-attention, one sequence asks questions of another. That distinction is the heart of Part B.

When HR attends to ECG, the model can ask:

- which ECG-HRV states match this HR pattern
- whether the current heart-rate token should trust a contemporaneous ECG shift
- whether the useful ECG evidence is slightly earlier than the current HR point

That learned alignment is encoded in the cross-attention matrix. The matrix is not just a diagnostic artifact. It is the mechanism by which the model decides how much one modality should influence another at each timestep.

## 5. Why the three fusion strategies behave differently

Early fusion assumes the model can safely mix modalities before it understands them. This is cheap and often works as a baseline, but it makes the first linear projection responsible for untangling signals with different scales and roles.

Late fusion assumes modalities should be understood separately and only combined after each encoder has formed a summary. This preserves modality-specific structure, but it prevents token-level interaction during representation learning.

Cross-attention tries to capture the best of both ideas: first let each modality build its own representation, then let the modalities communicate explicitly. That is the most plausible inductive bias for physiological signals that are related but not redundant.

## 6. Why modality type embeddings matter

Cross-attention only helps if the network knows which signal each token comes from. Modality type embeddings provide that identity information. They play the same role that segment embeddings play in language models: they tell the network whether a token belongs to HR, ECG-HRV, or EMG before any attention operation.

Without them, the model sees vectors and positions but loses the notion of signal provenance.

## 7. Why modality dropout is built into the training design

If the cross-modal model always sees every modality, it can become brittle and silently over-rely on one stream. Modality dropout prevents that by randomly removing ECG or EMG during training. This forces the network to maintain useful fallback behaviour and makes the later ablation study scientifically meaningful instead of purely distribution-shift driven.

## 8. What the ablation study is actually measuring

The ablation study asks a simple question with real scientific value:

- how much predictive signal does each modality contribute beyond the rest

If removing ECG barely changes performance, then the synthetic HRV features are not adding much beyond raw HR and glucose context. That is not a failed result. It means the additional modality is either redundant or too weakly informative at this temporal resolution.

If removing ECG increases error more than removing EMG, then the model is leaning more heavily on autonomic information than muscle-activity information. That is exactly the kind of mechanistic comparison Part B is meant to expose.

## 9. Why mixed precision matters here

Part B has multiple encoders and extra attention blocks, so activation memory is substantially higher than in Part A. Mixed precision reduces that pressure without changing the model definition. It is the practical enabling technique that keeps the three-model comparison feasible on a 6GB GPU.

## 10. What success looks like

The most informative Part B outcome is:

- synthetic features pass the correlation sanity checks
- all three models train stably within memory limits
- cross-attention outperforms early and late fusion, especially at 60 minutes
- ablation shows a measurable penalty when ECG or EMG is removed

That result would support the core thesis of Part B: token-level multimodal interaction adds value when the modalities capture related but not identical physiological dynamics.

