import argparse
import os
import pickle

import torch
import torchvision
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPTextModelWithProjection
from diffusers import AutoencoderKL
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms = torchvision.transforms.Compose([
  torchvision.transforms.Resize((256, 256)),
  torchvision.transforms.ToTensor()
])

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model_id = "openai/clip-vit-base-patch32"
clip_model = CLIPTextModelWithProjection.from_pretrained(model_id).to(device).eval()

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").eval().to(device)


def embed_text(text):
  # padding="max_length" (NOT padding=True): pickled prompts are always 77 tokens,
  # so the CFG null / sampling prompts must be too
  inputs = processor(text=text, return_tensors="pt", padding="max_length", truncation=True, max_length=77).to(device)
  with torch.inference_mode():
    outputs = clip_model(**inputs)
  embeds = outputs.last_hidden_state  # B,77,512
  return embeds


class TextImageDataset(Dataset):
  def __init__(self, dataset, transforms=None):
    self.dataset = dataset
    self.transforms = transforms
    self.dataset_iter = iter(self.dataset)

  def __len__(self):
    return int(1e10)

  def __getitem__(self, idx):
    try:
      item = next(self.dataset_iter)
    except StopIteration:
      self.dataset_iter = iter(self.dataset)
      item = next(self.dataset_iter)
    image = item['jpg']  # PIL image
    prompt = item['json']['prompt']  # text
    prompt = embed_text(prompt).squeeze(0)  # 1,77,512 -> 77,512, because dataloader wraps 1, anyways
    image = self.transforms(image) if self.transforms else image
    return image, prompt


def load_stream(path):
  with open(path, "rb") as f:
    while True:
      try:
        yield pickle.Unpickler(f).load()
      except EOFError:
        break


def build_index(path, index):
  # scan the pickle ONCE, record the byte offset of every record
  if os.path.exists(index):
    return torch.load(index)
  offsets = []
  with open(path, "rb") as f:
    while True:
      pos = f.tell()
      try:
        pickle.Unpickler(f).load()  # advances f past one record
      except EOFError:
        break
      offsets.append(pos)
  torch.save(offsets, index)
  return offsets


def main():
  parser = argparse.ArgumentParser(description="Download text-to-image-2M, encode to VAE latents + CLIP embeddings, pickle to disk")
  parser.add_argument("--samples", type=int, default=16384, help="number of samples to encode (default 16384)")
  parser.add_argument("--base-folder", type=str, default="./data/", help="output folder (default ./data/)")
  args = parser.parse_args()

  os.makedirs(args.base_folder, exist_ok=True)
  pickle_path = os.path.join(args.base_folder, "tensors.pkl")
  index_path = os.path.join(args.base_folder, "offsets.pt")

  ds = load_dataset("jackyhate/text-to-image-2M", streaming=True)
  ds_obj = TextImageDataset(ds['train'], transforms=transforms)
  dataloader = DataLoader(ds_obj, batch_size=1)
  dl_iter = iter(dataloader)

  with open(pickle_path, "wb") as f:
    pickler = pickle.Pickler(f, protocol=pickle.HIGHEST_PROTOCOL)
    for i in tqdm(range(args.samples)):
      next_obj = next(dl_iter)
      image = next_obj[0].to(device)
      image = image*2-1  # MUST RUN /2 + 0.5 .CLAMP (0,1)
      with torch.inference_mode():
        image = vae.encode(image).latent_dist.sample()*vae.config.scaling_factor
      # store on CPU: forked DataLoader workers can't deserialize CUDA tensors
      pickler.dump((image.cpu(), next_obj[1].cpu()))
      pickler.clear_memo()

  if os.path.exists(index_path):
    os.remove(index_path)  # byte offsets are now stale -> force rebuild
  build_index(pickle_path, index_path)

  sample = next(iter(load_stream(pickle_path)))
  print(f"wrote {args.samples} records to {pickle_path}")
  print(f"latent shape: {sample[0].shape}, prompt shape: {sample[1].shape}")


if __name__ == "__main__":
  main()
