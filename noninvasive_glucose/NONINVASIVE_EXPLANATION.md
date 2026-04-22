# Non-Invasive Glucose Estimation: Theory Driving the Architecture

## Purpose of this project

This project is not a variant of the supervised glucose forecasting pipeline. It solves a different problem with a different deployment constraint. In the supervised project, recent glucose was available as model input, so the Transformer could treat blood glucose itself as a primary state variable and forecast where it would go next. In this non-invasive project, glucose is never available at inference time. The model must estimate current glucose from biosignals alone, using glucose only during training as a teacher signal. That single change forces the rest of the system to change with it: the input window becomes shorter, the output becomes probabilistic, calibration becomes central, and encoder pretraining matters more because the signal-to-target relationship is weaker.

The design in this folder follows one guiding principle: use physiology to constrain the model. A non-invasive estimator cannot lean on the easiest shortcut, which would be "just read the most recent CGM value and extrapolate." It instead has to infer metabolic state from indirect correlates such as autonomic tone, exercise state, sleep stage, and cerebral perfusion. Those correlates are real, but they are noisy, delayed, and person-specific. The architecture therefore combines multimodal encoding, uncertainty estimation, and fast user-level calibration rather than pretending the biosignal-to-glucose mapping is deterministic and universal.

## The calibration paradox and its resolution

The central paradox of non-invasive glucose estimation is easy to state: if calibration requires glucose measurements, is the device really non-invasive? The resolution is to separate *training supervision* from *deployment sensing*. Continuous glucose monitoring is used here as a teacher, not as a permanent sensor modality. During development, CGM labels provide the target for learning how biosignals map onto metabolic state. During deployment, the model runs on biosignals alone and may only request a few sparse finger-prick or reference measurements during onboarding to adapt a user-specific offset.

That distinction is why the model outputs current glucose from biosignals only, and why the calibration module updates only a compact user representation rather than the full network. The encoders are meant to learn population-level physiology: how autonomic signatures, sleep, exercise, and perfusion correlate with glucose. Calibration then teaches the model who the new user is within that learned physiological space. In other words, calibration is not replacing the model; it is selecting the right point on the model's learned manifold.

This is also why the project uses a MAML-style inner loop at deployment time. With only a handful of calibration readings, updating every parameter would overfit instantly. Updating a 16-dimensional user embedding is a much better match to the information content of three sparse glucose labels. The support set should teach the model whether this user runs higher or lower, responds faster or slower, and shows stronger or weaker autonomic signatures than the population prior.

## The physiological signal chain: CBF to hypothalamus to autonomic output to glucose

The model includes CBF, HR/HRV, EEG, and EMG because they sit at different points in a biologically plausible control chain. Cerebral blood flow reflects global perfusion and can shift with posture, exercise, vascular tone, and metabolic demand. The hypothalamus integrates metabolic signals and participates in glucose regulation through autonomic and endocrine pathways. Autonomic output then shapes cardiovascular patterns that are visible in heart rate and HRV, while downstream behavioural and muscular state appears in EMG and movement-linked physiological changes. Glucose is therefore not treated as an isolated scalar but as the outcome of a coupled regulatory system.

The code follows that logic by making HR the central query stream during fusion. HR and HRV are the most directly and consistently observable autonomic markers among the chosen inputs, so the model lets the HR representation attend to ECG-derived HRV, EMG activity, EEG band structure, and CBF context. That choice is not a claim that HR causes glucose. It is a claim that HR is a strong hub signal through which multiple regulatory pathways are reflected. The fusion design uses HR as the anchor while still allowing other modalities to contribute context.

CBF is especially important conceptually even if it may not become the strongest predictive modality in practice. It is a slower signal than HR or EMG, so the model should not expect abrupt token-to-token changes. That is why the input window for this project is only 30 minutes and sampled at 5-minute resolution for slow signals: the estimator is focused on current state inference, not long-range temporal forecasting. CBF can contribute a coarse metabolic backdrop rather than a fast event trace.

## Why HRV correlates with glucose: the vagal-insulin pathway

Heart-rate variability is included because glucose regulation and autonomic regulation are coupled. In broad terms, parasympathetic and sympathetic balance influences insulin secretion, hepatic glucose output, and stress-related metabolic responses. HRV features such as HF power, RMSSD, and SDNN are therefore useful indirect markers of metabolic state, especially around meals, stress, and recovery periods.

The key point is not that any single HRV metric uniquely identifies glucose. It does not. The point is that HRV carries information about *state*: relaxed vagal dominance, sympathetic arousal, postprandial activation, recovery after exercise, or nocturnal stability. Those states change the probability distribution of glucose. That is exactly why the model should produce both a mean and an uncertainty estimate instead of a single hard prediction. Some HRV patterns constrain glucose tightly; others only weakly narrow the possibilities.

This physiological argument drives two code choices. First, ECG is represented as engineered HRV features rather than raw ECG waveforms. For a small-data, 6GB-VRAM-constrained non-invasive system, hand-compressed HRV features preserve the clinically meaningful autonomic summary while avoiding the memory cost of high-rate waveform attention. Second, ECG gets its own encoder and cross-attention path rather than being mixed blindly with other signals at the input. The model should first learn what an HRV sequence means before deciding how it should alter the glucose estimate.

## Why EEG sleep stages predict glucose

EEG is the most informative modality for overnight metabolic context because sleep stage modulates hormonal milieu. Deep sleep, light sleep, and wake transitions are associated with different patterns of cortisol, growth hormone, sympathetic tone, and insulin sensitivity. Overnight glucose regulation is therefore not just a cardiovascular story. If the model cannot recognise whether the user is in stable deep sleep, fragmented light sleep, or waking transition, it misses one of the strongest non-invasive signals about metabolic state.

This is why the project uses band-power EEG features instead of long raw EEG sequences. For the non-invasive estimation task, the information that matters most is not micro-morphology of waveforms but coarse state: delta-dominant deep sleep, theta/alpha transitions, or beta-dominant wakefulness. Relative band powers capture exactly that. They are cheap to compute, interpretable, and well matched to the 30-minute estimation window.

EEG also receives the most explicit pretraining strategy. On small glucose datasets, asking the model to discover sleep staging and glucose relevance simultaneously is inefficient. A much better strategy is to pretrain the EEG encoder on sleep-stage classification so it already knows how to map band-power patterns onto physiological state. Fine-tuning can then focus on the second mapping: physiological state to glucose. This staged learning matches the causal structure of the problem.

## Why EMG predicts post-exercise glucose

EMG matters because muscle is a direct glucose sink during and after exercise. This is a different type of relationship from HRV or EEG. HRV is mostly reflective of regulatory state; EMG is closer to the effector side of glucose disposal. When muscular activity rises, glucose uptake can increase through contraction-mediated pathways even before slower systemic responses fully settle.

That directness is why EMG is expected to dominate in post-exercise windows, especially when recent activity is still present in the 30-minute history. The model should be able to distinguish "heart rate is high because of stress" from "heart rate is high because of genuine muscular work" by consulting EMG. Without EMG, those situations are easier to confuse. This is also why the expected attribution patterns in the interpretability module are state-specific rather than fixed: EMG should matter a lot after exercise and much less during sedentary post-meal periods.

## Estimation versus forecasting: why the architecture changes

The supervised project forecast future glucose and had access to recent glucose context. This project estimates current glucose and has no such context. That changes three important architectural assumptions.

First, the history window becomes shorter. Forecasting future glucose benefits from long temporal context because the model must see delayed consequences of meals, exercise, and previous glucose excursions. Current-state estimation instead needs enough context to infer what metabolic regime the body is in now. Thirty minutes is a better default than two hours for that purpose because it preserves near-term autonomic and activity cues without spending capacity on stale history.

Second, the model output becomes probabilistic. In forecasting with glucose context, the immediate metabolic state is partially observed. In non-invasive estimation, it is latent. Multiple glucose values may be consistent with similar biosignal windows, especially across users. A single MSE-trained point estimate collapses that ambiguity into one number and silently hides uncertainty. A probabilistic head is therefore the right inductive bias.

Third, user-specific calibration becomes essential rather than optional. When glucose itself is not observed at inference time, baseline differences across users become much harder to disentangle from signal variation. The architecture therefore preserves a user embedding and a calibration procedure so deployment can adapt with only a few labelled points.

## Monte Carlo Dropout for uncertainty

The project uses Monte Carlo Dropout because uncertainty is not an add-on in a non-invasive estimator; it is part of the product requirement. If the model cannot tell when the biosignal evidence is weak or conflicting, the device cannot decide when to trust itself and when to defer. Dropout already exists as a regulariser during training. Keeping it active at inference and sampling multiple stochastic forward passes gives a cheap approximation to epistemic uncertainty without requiring a heavy Bayesian neural network.

This choice matches the hardware constraint. With 6GB VRAM and a multimodal Transformer, full Bayesian parameter inference is unrealistic. MC Dropout gives most of the practical value at a fraction of the implementation and compute cost. The model can report a mean estimate together with the spread across repeated stochastic passes. Wide spread indicates sensitivity to missing or ambiguous evidence. Narrow spread indicates consistent internal agreement.

The configuration therefore keeps dropout slightly higher than in the supervised project. That is deliberate. If dropout is too low, the Monte Carlo samples collapse toward each other and the epistemic estimate becomes uninformative. If it is too high, the base predictor becomes unstable. The chosen setting is a compromise between representational robustness and usable uncertainty sampling.

## Why negative log-likelihood is better than MSE here

The model's prediction head outputs a mean and a log-variance, and training uses Gaussian negative log-likelihood rather than MSE. This matters because the biosignal-to-glucose mapping has heteroscedastic noise. Some windows are intrinsically easier than others. Stable fasting periods with coherent autonomic signals should be predicted confidently. Post-exercise recovery or conflicting multimodal evidence should produce broader predictive uncertainty.

MSE treats every target as equally certain and punishes all residuals quadratically with no ability to say "this case is ambiguous." NLL instead learns to balance two pressures. The variance term penalises the model for claiming to be uncertain too often. The squared-error-over-variance term penalises it for being confidently wrong. The optimum is not merely accurate prediction; it is *honest* prediction. That is exactly the behaviour needed when the device may need to flag low-confidence estimates above a clinical threshold.

Using log-variance rather than raw variance is a stability choice. The network can output any real number, and exponentiation guarantees a positive variance. Clamping the log-variance prevents pathological numerical explosions during early training or outlier batches.

## Calibration through a MAML-style inner loop

The deployment calibration procedure borrows the inner-loop idea from MAML but applies it with a stricter constraint: only the user embedding is updated. That makes the adaptation step lightweight, data-efficient, and safe. The underlying encoders remain frozen, so the model keeps its shared physiological prior while the user embedding shifts the estimate toward the individual's baseline and response pattern.

The calibration readings are not chosen arbitrarily. A good calibration session samples distinct metabolic regimes: fasting, post-meal, and post-exercise. Three similar points would mostly teach the model one local offset. Three contrasting points constrain the embedding much better because they expose the user's baseline, excursion height, and recovery behaviour. In effect, the support set is designed to identify the user along the physiological axes the model already knows.

Using several inner-loop steps rather than one is justified here because deployment calibration is sparse but high value. With only a user embedding being updated, a few more gradient steps are cheap and can materially improve personalisation without risking catastrophic forgetting.

## Transfer learning for small datasets

A non-invasive glucose dataset is almost always too small to train every modality encoder from scratch. The solution is transfer learning with modality-specific pretraining tasks that align with the semantics of each signal. EEG can be pretrained on sleep staging. ECG can be pretrained on HRV structure or feature reconstruction. EMG can be pretrained on rest-versus-active discrimination. These tasks are easier than glucose estimation and use stronger local labels. They teach the encoders how to parse their modality before the full fusion model asks how those parsed states relate to glucose.

The fine-tuning strategy follows the same logic. The project freezes pretrained encoders for an initial phase so the fusion layers and uncertainty head can stabilise around strong modality representations. After that, the encoders are unfrozen with a lower learning rate so they can adapt without destroying their pretrained structure. This is especially important for EEG, where sleep-stage information is one of the most important indirect signals for overnight glucose estimation.

## Architecture summary implied by the theory

Putting the theory together yields a specific architecture. Each modality gets its own compact encoder because the physiological meaning of its tokens differs. HR is the fusion anchor because autonomic patterns are central and broadly available. ECG, EMG, EEG band powers, and CBF contribute through cross-attention into the HR representation. A learned user embedding is appended as personal context. A final Transformer layer integrates the fused sequence. The prediction head outputs both a mean glucose estimate and log-variance. Training uses NLL. Inference uses MC Dropout. Deployment calibration updates only the user embedding with a few sparse labels.

Every major code decision in this folder follows from the biological and deployment constraints above. The system is not trying to outperform a CGM-assisted forecasting model on pure RMSE. It is trying to achieve the best possible current-glucose estimate from non-invasive signals alone while knowing when it does and does not know enough. That is the correct goal for this problem.
