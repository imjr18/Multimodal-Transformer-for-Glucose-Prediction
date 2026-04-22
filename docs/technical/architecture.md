# Architecture Guide

This document describes the actual model families in the repository and the tensor shapes they operate on. `B` denotes batch size.

## Part A — `TemporalTransformer`

Configuration:

- `d_model = 64`
- `n_heads = 4`
- `n_encoder_layers = 2`
- `d_ff = 256`
- `input_len = 24`
- `target_offsets = [6, 12]`

Signal flow:

```text
HR Input [B, 24, 1]
       │
       ├── TokenEmbedding(1 -> 64)
       │        ↓
       │   [B, 24, 64]
       │
Glucose Context [B, 24, 1]
       │
       ├── TokenEmbedding(1 -> 64)
       │        ↓
       │   [B, 24, 64]
       │
       └── Concatenate along feature dim
                ↓
           [B, 24, 128]
                │
                ├── Linear(128 -> 64)
                │        ↓
                │   [B, 24, 64]
                │
                ├── Prepend learnable CLS token
                │        ↓
                │   [B, 25, 64]
                │
                ├── Sinusoidal positional encoding
                │
                ├── TransformerEncoder × 2
                │   ┌─────────────────────────────┐
                │   │ MultiHeadAttention          │
                │   │ 4 heads, d_k = 16 each      │
                │   │ FFN: 64 -> 256 -> 64        │
                │   │ Residual + Pre-LN           │
                │   └─────────────────────────────┘
                │
                ├── Take CLS output only
                │        ↓
                │   [B, 64]
                │
                └── RegressionHead
                    Linear(64,64) -> ReLU -> Dropout -> Linear(64,2)
                         ↓
                    [B, 2]
```

Outputs:

- index `0`: glucose at `t+30` minutes
- index `1`: glucose at `t+60` minutes

Model size:

- `TemporalTransformer`: `112,834` trainable parameters

## Part B — Multimodal fusion

### Early fusion

Inputs at each timestep:

- HR: `1`
- glucose context: `1`
- ECG-HRV: `5`
- EMG: `2`

Total features per timestep: `9`

```text
[B, 24, 9]
   │
   ├── Linear(9 -> 64)
   ├── Positional encoding
   ├── CLS token
   ├── TransformerEncoder × 2
   └── RegressionHead -> [B, 2]
```

Model size:

- `EarlyFusionTransformer`: `104,962` parameters

### Late fusion

Three encoders run independently:

- HR branch sees `[B, 24, 2]` because HR and glucose context are concatenated
- ECG branch sees `[B, 24, 5]`
- EMG branch sees `[B, 24, 2]`

Each branch emits a CLS summary `[B, 64]`, then:

```text
[B, 64] + [B, 64] + [B, 64]
              │
              └── Concatenate -> [B, 192]
                        │
                        └── FusionHead: 192 -> 64 -> 2
```

Model size:

- `LateFusionTransformer`: `313,346` parameters

### Cross-attention fusion

Each modality uses a dedicated encoder that keeps a leading CLS token, so branch outputs are `[B, 25, 64]`.

```text
HR + glucose context [B, 24, 2] -> HR encoder  -> [B, 25, 64]
ECG-HRV            [B, 24, 5] -> ECG encoder -> [B, 25, 64]
EMG                [B, 24, 2] -> EMG encoder -> [B, 25, 64]

HR attends to ECG: query=HR, key=value=ECG
HR attends to EMG: query=HR, key=value=EMG

Residual-style fusion:
hr_fused = LayerNorm(hr + hr_from_ecg + hr_from_emg)

Take CLS token from hr_fused -> RegressionHead -> [B, 2]
```

Model size:

- `CrossModalTransformer`: `338,754` parameters

## Part C — `FullModalTransformer`

Part C reuses the HR-centric Part B fusion path and adds EEG plus CBF.

Common branch shapes:

- HR + glucose context: `[B, 24, 2] -> [B, 25, 64]`
- ECG-HRV: `[B, 24, 5] -> [B, 25, 64]`
- EMG: `[B, 24, 2] -> [B, 25, 64]`
- CBF: `[B, 24, 1] -> [B, 25, 64]`

### EEG variants

#### Frequency EEG

```text
Raw 2-minute EEG -> 120 one-second segments
                -> 5 relative band powers per second
                -> [B, 120, 5]
                -> encoder -> summary [B, 64]
```

#### Patch EEG

```text
Raw EEG [B, 30720]
      -> reshape into 480 patches of 64 samples
      -> [B, 480, 64]
      -> linear projection to model space
      -> Transformer encoder
      -> pooled summary [B, 64]
```

#### Hierarchical EEG

```text
Raw EEG [B, 30720]
      -> 24 windows of 5 seconds
      -> each local window: 20 patches of 64 samples
      -> local summary [B, 24, 32]
      -> project to [B, 24, 64]
      -> global encoder
      -> summary [B, 64]
```

### Full fusion path

```text
HR branch  [B, 25, 64]
ECG branch [B, 25, 64]
EMG branch [B, 25, 64]
CBF branch [B, 25, 64] -> take CLS summary [B, 64] -> broadcast [B, 25, 64]
EEG summary [B, 64] -> project -> [B, 1, 64]

HR attends to ECG / EMG / CBF
        │
        └── hr_fused [B, 25, 64]
                 │
                 └── append EEG summary token
                          ↓
                     [B, 26, 64]
                          │
                          └── final fusion encoder (1 layer)
                                   │
                                   └── CLS -> RegressionHead -> [B, 2]
```

Model sizes:

- `FullModalTransformer` with `frequency_eeg`: `610,178`
- `FullModalTransformer` with `patch_eeg`: `613,954`
- `FullModalTransformer` with `hierarchical_eeg`: `626,754`

## Part D — user conditioning and meta-learning wrapper

Part D does not replace the Part C backbone. It wraps it with user-specific conditioning.

Conceptual tensor operation at each conditioned layer:

```text
CLS token           [B, 1, 64]
User embedding      [B, 16]
Expanded user token [B, 1, 16]
Concatenate         [B, 1, 80]
Project back        [B, 1, 64]
Replace conditioned CLS before attention
```

The rest of the sequence remains unchanged. This keeps the conditioning cheap because only the sequence summary token is modified.

## Non-Invasive — `NonInvasiveTransformer`

Configuration highlights:

- `window_timesteps = 6`
- `d_model = 64`
- `user_emb_dim = 16`
- final output is `(mean, log_var)`

Per-modality shapes:

- HR: `[B, 6, 1] -> [B, 6, 64]`
- ECG-HRV: `[B, 6, 5] -> [B, 6, 64]`
- EMG: `[B, 6, 2] -> [B, 6, 64]`
- EEG bands: `[B, 6, 5] -> [B, 6, 64]`
- CBF: `[B, 6, 1] -> [B, 6, 64]`

Fusion and prediction:

```text
HR encoded   [B, 6, 64]
ECG encoded  [B, 6, 64]
EMG encoded  [B, 6, 64]
EEG encoded  [B, 6, 64]
CBF encoded  [B, 6, 64]
     │
     └── HR-centric cross-attention fusion
              ↓
         fused_hr [B, 6, 64]

User embedding [B, 16] -> Linear(16 -> 64) -> user token [B, 1, 64]
CLS token      [B, 1, 64]

Concatenate [CLS, fused_hr, user_token]
      ↓
 [B, 8, 64]
      │
      └── final TransformerEncoder (1 layer)
               │
               └── take CLS -> UncertaintyHead
                        │
                        ├── mean    [B]
                        └── log_var [B]
```

Model size:

- `NonInvasiveTransformer`: `627,394` parameters

## Why the architectures stay small

All of these designs are constrained by the same 6GB VRAM target. The repository therefore keeps:

- `d_model=64`
- `n_heads=4`
- shallow encoder stacks
- aggressive sequence reduction for expensive modalities
- mixed precision where multimodal activations would otherwise dominate memory

That constraint is not incidental. It is one of the main design drivers of the entire repository.

