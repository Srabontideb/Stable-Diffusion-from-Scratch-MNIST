# Stable Diffusion from Scratch — MNIST

A clean, fully commented implementation of **Denoising Diffusion Probabilistic Models (DDPM)** trained on MNIST. Built for learning — every line of code maps directly to the math in the original paper.

---

## What this teaches you

| Concept | Where in code |
|---|---|
| Forward noising process | `NoiseSchedule.q_sample()` |
| Linear β noise schedule | `NoiseSchedule.__init__()` |
| Sinusoidal time embedding | `SinusoidalTimeEmbedding` |
| Residual block with time injection | `ResBlock` |
| Self-attention at bottleneck | `SelfAttention` |
| U-Net encoder / decoder | `UNet.forward()` |
| DDPM training loop | `train()` |
| DDPM reverse sampling | `sample_ddpm()` |

---

## How it works

Diffusion models have two processes:

```
Forward  →  x₀ ──(add noise)──► x₁ ──► x₂ ──► ... ──► xₜ  (pure noise)
Reverse  ←  x̂₀ ◄──(denoise)── x₁ ◄── x₂ ◄── ... ◄── xₜ  (learned by U-Net)
```

**Training:** Pick a random timestep `t`, corrupt a real image with Gaussian noise, ask the U-Net to predict that noise. Minimise MSE.

**Sampling:** Start from pure Gaussian noise, run the U-Net T times in reverse, each step removing a little noise until a generated image emerges.

### Core equations

```
Forward  (closed form):  xₜ = √ᾱₜ · x₀  +  √(1−ᾱₜ) · ε      ε ~ N(0,I)
Training loss:           L  = ||ε  −  ε̂θ(xₜ, t)||²
Reverse  update:         xₜ₋₁ = (1/√αₜ)(xₜ − βₜ/√(1−ᾱₜ) · ε̂)  +  √βₜ · z
```

---

## Project structure

```
stable_diffusion_tutorial.py   ← single-file implementation
│
├── NoiseSchedule              ← β schedule + forward process q(xₜ|x₀)
├── SinusoidalTimeEmbedding    ← encodes timestep t into a vector
├── ResBlock                   ← conv block that injects time embedding
├── SelfAttention              ← spatial attention at bottleneck
├── UNet                       ← full denoiser (encoder + bottleneck + decoder)
│
├── get_dataloader()           ← MNIST, normalised to [−1, 1]
├── train()                    ← DDPM training algorithm
├── sample_ddpm()              ← DDPM reverse sampling
│
└── visualisation helpers
    ├── show_forward_process()     → forward_process.png
    ├── show_loss_curve()          → loss_curve.png
    ├── show_generated_samples()   → generated_samples.png
    └── show_denoising_trajectory()→ denoising_trajectory.png
```

---

## Quick start

### Prerequisites

```bash
pip install torch torchvision matplotlib numpy tqdm
```

A CUDA-capable GPU is recommended but not required. On CPU, expect ~45 min per epoch; on a T4 GPU, ~3 min per epoch.

### Run

```bash
python stable_diffusion_tutorial.py
```

On first run, MNIST (~11 MB) downloads automatically into `./data/`.

### Colab (recommended for beginners)

1. Open [colab.research.google.com](https://colab.research.google.com)
2. Upload `stable_diffusion_colab.ipynb`
3. **Runtime → Change runtime type → T4 GPU**
4. Run All (`Ctrl+F9`)

---

## Configuration

All hyperparameters are at the top of the file:

```python
IMG_SIZE   = 28     # image resolution (MNIST is 28×28)
CHANNELS   = 1      # 1 = grayscale, 3 = RGB
BATCH_SIZE = 128
EPOCHS     = 30     
LR         = 3e-4   # Adam learning rate
T          = 300    # total diffusion timesteps
```

---

## Outputs

After training completes, four PNG files are saved:

| File | Shows |
|---|---|
| `forward_process.png` | One image progressively noised from t=0 to t=T |
| `loss_curve.png` | Training MSE loss over epochs |
| `generated_samples.png` | 4×4 grid of generated digits |
| `denoising_trajectory.png` | Single sample going from noise → image step by step |

The trained model is saved as `unet_mnist.pth`.

---

## Architecture overview

```
Input: (B, 1, 28, 28) noisy image  +  (B,) timestep t
                          │
              SinusoidalTimeEmbedding(t)
              → (B, 128) time vector, injected into every ResBlock
                          │
           ┌──── Encoder ─────────────────────────────┐
           │  enc_in → ResBlock → 28×28 @ 64ch  (x1)  │
           │  Conv↓  → ResBlock → 14×14 @ 128ch (x2)  │
           │  Conv↓  →           7×7  @ 256ch  (x3)  │
           └──────────────────────────────────────────┘
                          │
           ┌──── Bottleneck ──────────────────────────┐
           │  ResBlock → Self-Attention → ResBlock     │
           │             7×7 @ 256ch                  │
           └──────────────────────────────────────────┘
                          │
           ┌──── Decoder (with skip connections) ──────┐
           │  ConvUp + cat(x2) → ResBlock → 14×14      │
           │  ConvUp + cat(x1) → ResBlock → 28×28      │
           └──────────────────────────────────────────┘
                          │
Output: (B, 1, 28, 28) predicted noise ε̂
```

**Skip connections** pass encoder features directly into the decoder so fine spatial detail is not lost at the bottleneck.

**Time injection** — each `ResBlock` adds the projected time vector to its feature map, so every layer knows the current noise level.

---

## Extending the code

### Faster sampling (DDIM — no retraining needed)

Replace the sampling loop with a deterministic 50-step update instead of the full 300-step stochastic process.

```python
# 10× faster, same model weights
samples = sample_ddim(model, schedule, steps=50)
```

### Better noise schedule (cosine)

```python
T          = 500
# Change in NoiseSchedule.__init__:
self.betas = cosine_schedule(T)   # instead of linspace
```

### Colour images (CIFAR-10)

```python
IMG_SIZE = 32
CHANNELS = 3
# Change dataset to datasets.CIFAR10(...)
```

### Class-conditional generation

```python
# Add to UNet: self.class_emb = nn.Embedding(10, 128)
# In forward:  t_emb = t_emb + self.class_emb(y)
# At sampling: pass y=torch.full((16,), fill_value=7)  # generate sevens
```

### EMA weights (free quality improvement)

```python
from copy import deepcopy
ema_model = deepcopy(model)
# After each optimizer.step():
for p_e, p in zip(ema_model.parameters(), model.parameters()):
    p_e.data = 0.9999 * p_e.data + 0.0001 * p.data
# Use ema_model for sampling
```

---


## Paper reference

Ho, J., Jain, A., & Abbeel, P. (2020). **Denoising Diffusion Probabilistic Models**.
[arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)

---

## License

MIT — free to use, modify, and learn from.
