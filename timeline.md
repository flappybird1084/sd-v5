# Project Timeline

A chronological account of how this rectified-flow text-to-image model evolved, reconstructed from file modification times and the code itself. The single git commit (`b51df53 "sorta working v3"`) bundles everything, so ordering below comes from filesystem timestamps and the lineage visible in the code.

Throughout, the architecture is the same idea: a **convolutional autoencoder** compresses a 2D image into a sequence of latent tokens, a stack of **joint-attention transformer blocks** ("blender") denoises that latent conditioned on a prompt + timestep, and training uses **rectified flow** (a.k.a. flow matching) — predict the velocity `x1 - x0` along a straight line between noise `x0` and data `x1`, then integrate it at inference. Classifier-free guidance (CFG) is used for conditioning.

| Order | Artifact | Last modified | Role |
|-------|----------|---------------|------|
| 1 | `rf-v3.ipynb` | Jun 4 19:37 | MNIST prototype (proof of concept) |
| 2 | `t2i-v1.ipynb` | Jun 4 23:18 | First text-to-image port |
| 3 | `components/model.py` | Jun 7 12:14 | Extracted/refactored model for t2i |
| 4 | `t2i-v2.ipynb` | Jun 7 22:14 | Scale-up + CFG + bf16 + compile |
| 5 | `components/model2.py` | Jun 8 11:05 | Reworked latent layout & projections |
| 6 | `t2i-v3.ipynb` | Jun 8 15:21 | Current working version (uses `model2`) |
| 7 | `t2i-v4.ipynb` | Jun 8 15:58 | Empty placeholder (next iteration) |

---

## 1. `rf-v3.ipynb` — MNIST prototype (Jun 4)

Despite the "v3" in the name, this is the **earliest** file and the conceptual seed of everything else. It is self-contained (defines its model inline, no `components/` import) and trains on **MNIST at 32×32**, using the digit label as the "prompt."

Key pieces, all defined in the notebook:

- **`Encoder`** — 3 conv + maxpool stages, `1→32→64→dim` channels, downsampling 32×32 → 16×16. Flattens to `B,256,dim` (256 spatial tokens) and `LayerNorm`s over `dim`. Note the **`B, T, C` token layout** (tokens in the sequence dim) — this convention carries forward.
- **`BlenderInternals`** — the joint-attention block. Embeds the prompt (an `nn.Embedding(11, dim)`: digits 0–9 plus index 10 = the null/unconditional token), builds sinusoidal **timestep embeddings** (`timestep_fc` projection, `*1000` scaling), adds learned **positional embeddings** to the image tokens, then runs a single `scaled_dot_product_attention` over the **concatenation of image + prompt tokens** (joint attention), followed by a GELU MLP with residual + LayerNorm. Returns updated image and prompt tokens.
- **`Decoder`** — `ConvTranspose2d` stack mirroring the encoder, `B,L,D → B,D,H,W → image`.
- **`BlenderV2`** — wraps `encoder`, a `blenderhead` block, `num_layers-1` more `blenderinternals` blocks, and `decoder`. Has both `forward` (image→latent→denoise) and **`forward_latent`** (operate directly on a latent `z`, used in training/inference). Residual connections wrap every block.

**Training recipe (the template reused everywhere after):**
1. Train the autoencoder alone with MSE reconstruction loss (`autoencoder_epochs`).
2. **Freeze** encoder + decoder weights.
3. Train rectified flow on the **frozen latent space**: sample `t`, form `x_t = x0*(1-t) + x1*t`, predict velocity with `forward_latent`, regress to `v_target = x1 - x0`.
4. CFG dropout: with prob `drop_prob=0.1`, replace the prompt with the null label (`torch.full_like(prompt, 10)`).
5. bf16 autocast + `torch.compile`.

Inference does Euler integration of the velocity field in latent space, then decodes once at the end (with `guidance_scale=5.0`).

This notebook ends in load-only mode (`load_model=True, train_model=False`), i.e. it was a working checkpoint.

## 2. `t2i-v1.ipynb` — first text-to-image port (Jun 4, later same day)

Generalizes the MNIST prototype to **real text-to-image** at **256×256**.

- **Data:** streams `jackyhate/text-to-image-2M` (HF, `streaming=True`). `ImageDataset` yields images for AE training; `TextImageDataset` yields `(image, prompt_string)` pairs. An infinite `__len__` + iterator-reset pattern handles the streaming dataset.
- **Text conditioning:** swaps the integer-label embedding for **CLIP text embeddings** (`openai/clip-vit-base-patch32`, `CLIPTextModelWithProjection`). `embed_text` produces a `B,1,512` projection, which is **split into 2 tokens of 256** (`torch.chunk` then `cat`) → `B,2,256`, so the prompt is 2 tokens at `d_model=256`.
- **Model:** now imported from `components/model.py` (`from components.model import DiffusionModel`) rather than defined inline.
- Same two-phase recipe (train AE → freeze → train RF). Saves to `checkpoints/model-v1.pth`.

This version has **no CFG dropout yet** in the training loop and **no fp16/compile** — it's the minimal "does the port even run" pass. `rf_steps = 10000`.

## 3. `components/model.py` — extracted t2i model (Jun 7)

The model factored out of the notebook into a reusable module: `Encoder`, `Decoder`, `JointAttention`, `DiffusionModel`.

- Operates at **`d_model=256`**. Encoder produces `B,512,256` — i.e. **512 tokens** (the 16×16=256 spatial positions × ... ) wait, concretely: conv stack → `B,512,16,16` → `view` to `B,512,256`. So here the **channel dim (512) is the sequence length** and 256 (=16×16 flattened) is the feature dim. (This layout is exactly what `model2.py` later reverses — see below.)
- `JointAttention` adds timestep embedding, a learned `nn.Embedding(512, dim)` positional embedding, then joint attention over `image (512 tok) + prompt (2 tok) = 514` tokens, MLP + 2× LayerNorm with residual.
- `DiffusionModel` bundles CLIP processor + text model, `embed_text` (splits CLIP `512` into `2×256` to match the prompt-token convention), `encode`/`decode`, and a `forward` that runs `num_layers` joint-attention blocks with running residuals and returns the image velocity prediction.

This is the v1/v2 model. The `__main__` smoke test runs a single `1,3,256,256` image + `"a cat on a skateboard"` end to end.

## 4. `t2i-v2.ipynb` — scale-up & training hardening (Jun 7)

Same `components/model.py` model, but a serious training pass:

- **New dataset:** `MohamedRashad/midjourney-detailed-prompts` (uses `image` + `short_prompt` fields). Non-streaming.
- **Bigger model:** `DiffusionModel(num_layers=16)` (up from the default 4).
- **Bigger batch / throughput:** `batch_size=64`, `num_workers=12`, `prefetch_factor=8`, `persistent_workers`.
- **Classifier-free guidance added to training:** `cfg_prob=0.1`, with a precomputed `cfg_prompt = embed_text(['null']).repeat(batch_size,1,1)`; randomly swap the batch's prompt for the null embedding.
- **Mixed precision + compile:** `torch.compile(model)` and `torch.autocast(bfloat16)` around the forward/loss.
- `rf_steps = 100000`.
- **Inference cell added:** Euler-integrate with CFG (`v = v_null + scale*(v_pred - v_null)`), `guidance_scale=5`, 50 steps, decode and `plt.imshow`. Latent shape `[1,512,256]` matches `model.py`'s token layout. Test prompt: `"a red circle"`.

## 5. `components/model2.py` — reworked latent layout (Jun 8 morning)

A targeted rewrite of `model.py` addressing the **token/feature axis** of the latent, plus the prompt representation.

- **Transposed latent layout:** encoder now does `view(B,512,256).permute(0,2,1)` → **`B,256,512`**. The inline comment explains the reasoning: *"512 stacks → each pixel in a stack has its own patch position, thus those pixels should be in the T (sequence) dim of the transformer."* So now there are **256 sequence tokens** with feature dim 512, and `LayerNorm([256,512])`. The decoder permutes back before the deconv stack.
- **Single prompt token:** `embed_text` no longer splits the CLIP vector — it keeps `B,1,512` (1 prompt token at 512), with `prompt_tokens=1`.
- **Projections in/out of `d_model`:** since latent feature dim (512) ≠ working `d_model` (256), `JointAttention` gains `image_proj`/`prompt_proj` (512→dim) on the way in and `image_up`/`prompt_up` (dim→512) on the way out. Positional embedding is now `nn.Embedding(image_tokens=256, dim)`. Joint attention runs over `256 + 1 = 257` tokens.
- `DiffusionModel.forward` returns only the image velocity (and the `__main__` smoke test wraps inference in `torch.no_grad`).

## 6. `t2i-v3.ipynb` — current working version (Jun 8 afternoon)

Structurally almost identical to `t2i-v2.ipynb`, with the key change being the model swap:

- **`from components.model2 import DiffusionModel`** (the reworked layout).
- Same Midjourney dataset, `num_layers=16`, `batch_size=64`, CFG (`cfg_prob=0.1`), bf16 autocast, `torch.compile`, `rf_steps=100000`.
- `embed_text` keeps the **single `B,1,512` token** (the chunk/split is commented out, matching `model2`).
- **Inference latent shape updated to `[1,256,512]`** to match `model2`'s transposed layout. `guidance_scale=2`, richer test prompt (*"A serene white wolf, its fur a rainbow kaleidoscope..."*).
- Adds a **parameter-count cell** (`sum(p.numel())/1e6` m parameters).

This is the "sorta working v3" referenced in the commit message.

## 7. `t2i-v4.ipynb` — placeholder (Jun 8, latest)

Empty file (0 bytes), created right after v3 — the staging ground for the next iteration. Untracked in git.

---

## Lineage at a glance

```
rf-v3.ipynb            MNIST, inline model, B,T,C tokens, label-embedding prompt
   │  (port to 256x256 real images + CLIP text)
   ▼
t2i-v1.ipynb  ──uses──▶ components/model.py   (d_model=256, 512 seq tokens, 2 prompt tokens)
   │  (scale up: 16 layers, CFG, bf16, compile, midjourney data)
   ▼
t2i-v2.ipynb  ──uses──▶ components/model.py
   │  (rethink latent axis + single prompt token + in/out projections)
   ▼
t2i-v3.ipynb  ──uses──▶ components/model2.py  (256 seq tokens × 512 feat, 1 prompt token)
   │
   ▼
t2i-v4.ipynb           (empty — next)
```

**Recurring design constants across all versions**
- Two-phase training: reconstruction-pretrain the autoencoder, freeze it, then train rectified flow in the frozen latent space.
- Rectified-flow objective: `x_t = (1-t)·x0 + t·x1`, predict `v = x1 - x0`, MSE loss.
- Joint attention: image and prompt tokens are concatenated and attended together in a single SDPA call, per block, with residuals + LayerNorm + MLP.
- Classifier-free guidance: null-prompt dropout during training (~v2 onward), guided velocity at inference.
- Sinusoidal timestep embedding scaled by `1000`, added to tokens; learned positional embeddings on image tokens.

**What actually changed between versions**
- `rf-v3 → t2i-v1`: label embedding → CLIP text embeddings; 32×32 MNIST → 256×256 real images; inline model → `components/`.
- `t2i-v1 → t2i-v2`: +CFG, +bf16/compile, +bigger model & batch, dataset swap, +inference cell.
- `t2i-v2 → t2i-v3`: `model.py` → `model2.py`: latent transposed to `B,256,512`, single 512-d prompt token, added in/out linear projections to/from `d_model`.
```
