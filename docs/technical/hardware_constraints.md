# Hardware Constraints Decision Log

Every major design choice in this repository was shaped by a 6GB VRAM development target. This table explains what was chosen, what would likely be larger in an unconstrained setting, why the constrained choice is still scientifically useful, and what it costs.

| Area | Chosen Value | Larger-Unconstrained Alternative | Why the chosen value is valid | What is lost |
|------|--------------|----------------------------------|-------------------------------|--------------|
| Core model width | `d_model=64` | `128` or `256` | Enough capacity for small multimodal experiments while keeping attention and FFN activations cheap | Less representational headroom |
| Attention heads | `4` | `8` or more | Clean `64 / 4 = 16` per head and low overhead | Fewer attention subspaces |
| Encoder depth | `2` layers in most branches | `4-12` layers | Keeps optimisation and memory manageable on small datasets | Shallower feature hierarchy |
| FFN width | `256` | `512-1024` | Standard `4x` expansion ratio still holds | Reduced nonlinear capacity |
| Part A batch size | `32` | `64+` | Fits short supervised sequences safely | Slower epoch throughput |
| Part B batch size | `16` | `32+` | Multiple encoders and cross-attention increase activation memory | More noisy gradients |
| Part C batch size | `8` | `16+` | EEG makes activations much heavier | Longer training wall-clock time |
| Part C grad accumulation | `4` | `1` | Recovers effective batch size without holding all examples in memory | More optimizer-step overhead |
| EEG raw window | `2` minutes | `10+` minutes | Enough to capture sleep-state information while staying tractable | Loses longer uninterrupted context |
| Frequency EEG tokens | `120` | much larger raw-token streams | Strong sleep-stage summary at tiny cost | Loses waveform morphology |
| Patch EEG size | `64` samples | smaller patches or raw samples | `480` patches is still feasible | Coarser within-patch resolution |
| Hierarchical local model width | `32` | `64+` | Keeps the local encoder cheap because it runs 24 times | Less local expressivity |
| Non-invasive window | `30` minutes / `6` steps | `2` hours / `24` steps | Current-state estimation depends more on recent state than long history | Less long-lag temporal evidence |
| Non-invasive dropout | `0.15` | `0.1` or lower | Makes MC Dropout uncertainty more informative | Slightly noisier deterministic predictions |
| User embedding size | `16` | `32-64` | Enough room for baseline and response-style offsets | Lower personalisation capacity |
| MAML inner steps | `1` in supervised Part D | `5+` | First-order MAML is already expensive with the Part C backbone | Less expressive adaptation |
| Calibration steps | `3` in non-invasive | `5+` | More than Part D, but still cheap because only the user embedding updates | Limited refinement from sparse labels |
| Mixed precision | enabled where useful | full float32 everywhere | Large activation savings in multimodal training | Slightly more complex training loop |

## What would change with more VRAM

With a substantially larger GPU budget, the most immediate upgrades would be:

- deeper modality encoders
- wider `d_model`
- longer EEG windows or less aggressive EEG compression
- larger batch sizes for more stable optimisation
- second-order MAML rather than first-order approximation

Those changes would increase flexibility, but they would also make it harder to understand which modeling idea produced which gain. The constrained versions in this repository are therefore not only cheaper; they are also easier to reason about.

