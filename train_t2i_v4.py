"""Train the t2i v4 rectified-flow diffusion model.

Trains only: assumes the latent/text-embedding dataset already exists at
DATA_PATH (built by the t2i-v4 notebook's pickle-population cell). No data
prep, no sampling/inference here.
"""

import pickle

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPTextModelWithProjection
from tqdm import tqdm

from components.model3 import DiffusionModel

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
DATA_PATH = "data/tensors.pkl"
OFFSETS_PATH = "data/offsets.pt"
CHECKPOINT_PATH = "checkpoints/model-v3.pth"

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

# Model
IMAGE_TOKENS = 256        # 2x2 latent patches: (4,32,32) -> 256 tokens of 16
DIM = 1024
NUM_LAYERS = 16
NHEAD = 16

# Optimisation
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
RF_STEPS = 300_000
CFG_PROB = 0.1            # probability of dropping the prompt (classifier-free guidance)

# DataLoader
NUM_WORKERS = 8
PREFETCH_FACTOR = 4

COMPILE = True           # torch.compile the model
LOG_EVERY = 20
SAVE_EVERY = 10_000       # periodic checkpoint inside the loop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Text embedding (only used to build the null/CFG prompt)
# ---------------------------------------------------------------------------
processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
clip_model = CLIPTextModelWithProjection.from_pretrained(CLIP_MODEL_ID).to(device).eval()


def embed_text(text):
    # padding="max_length": every prompt (incl. the CFG null) must be exactly 77
    # tokens so cond/uncond batches always have the same shape
    inputs = processor(text=list(text), return_tensors="pt", padding="max_length",
                       truncation=True, max_length=77).to(device)
    with torch.inference_mode():
        outputs = clip_model(**inputs)
    return outputs.last_hidden_state  # B,77,512 — per-token, not pooled


# ---------------------------------------------------------------------------
# Dataset: pre-encoded VAE latents + text embeddings, read by byte offset
# ---------------------------------------------------------------------------
def build_index(path=DATA_PATH, index=OFFSETS_PATH):
    import os
    if os.path.exists(index):
        return torch.load(index)
    offsets = []
    with open(path, "rb") as f:
        while True:
            pos = f.tell()
            try:
                pickle.Unpickler(f).load()
            except EOFError:
                break
            offsets.append(pos)
    torch.save(offsets, index)
    return offsets


class PickledImageTextDataset(Dataset):
    def __init__(self, path=DATA_PATH):
        self.path = path
        self.offsets = build_index(path)
        self.f = None  # opened lazily, once per worker

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        if self.f is None:
            self.f = open(self.path, "rb")  # each worker gets its own handle
        self.f.seek(self.offsets[idx])
        data = pickle.load(self.f)

        # .detach(): stored latents kept a grad graph from vae.encode(); collate
        # with pin_memory uses torch.stack(out=...) which rejects grad tensors
        # records are (1,4,32,32) latent + (1,77,512) prompt hidden states
        image = data[0].squeeze(0).detach()
        prompt = data[1].squeeze(0).detach()
        return image, prompt


def main():
    dataset = PickledImageTextDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=NUM_WORKERS, persistent_workers=True,
                            prefetch_factor=PREFETCH_FACTOR, pin_memory=True)

    model = DiffusionModel(image_tokens=IMAGE_TOKENS, dim=DIM,
                           num_layers=NUM_LAYERS, nhead=NHEAD).to(device)
    print(f"{sum(p.numel() for p in model.parameters()) / 1_000_000:.1f}m parameters")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    cfg_prompt = embed_text(['']).to(device).repeat(BATCH_SIZE, 1, 1)
    train_model = torch.compile(model) if COMPILE else model

    pbar = tqdm(range(RF_STEPS))
    data_iter = iter(dataloader)

    for step in pbar:
        try:
            images, prompts = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            images, prompts = next(data_iter)

        images = images.to(device)
        prompts = prompts.to(device)

        # classifier-free guidance: occasionally drop the prompt
        if torch.rand(1) < CFG_PROB:
            prompts = cfg_prompt[:images.size(0)]

        t = torch.rand([images.size(0), 1, 1], device=device)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # patchify: (B,4,32,32) -> 2x2 spatial patches, all 4 channels per token
            x1 = images.reshape(images.size(0), 4, 16, 2, 16, 2)
            x1 = x1.permute(0, 2, 4, 1, 3, 5).reshape(images.size(0), IMAGE_TOKENS, -1)
            x0 = torch.randn_like(x1)
            x_t = x0 * (1 - t) + x1 * t

            predicted_velocity = train_model(x_t, prompts, t)
            target_velocity = x1 - x0

        loss = F.mse_loss(predicted_velocity, target_velocity)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % LOG_EVERY == 0:
            pbar.set_description(f"loss: {loss.item():.4f}")

        if step > 0 and step % SAVE_EVERY == 0:
            torch.save(model.state_dict(), CHECKPOINT_PATH)

    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"saved checkpoint to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
