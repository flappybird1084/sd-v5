import argparse
import os

import gradio as gr
import torch
from diffusers import AutoencoderKL
from huggingface_hub import hf_hub_download
from transformers import CLIPProcessor, CLIPTextModelWithProjection

from components.model3 import DiffusionModel

HF_REPO_ID = "flappybird1084/sd-v4"
HF_CHECKPOINT = "model-v3.pth"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint",
    default=f"checkpoints/{HF_CHECKPOINT}",
    help=f"path to .pth state_dict (downloaded from {HF_REPO_ID} if not found locally)",
)
parser.add_argument(
    "--device",
    choices=["cpu", "cuda"],
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="device to run on (default: cuda if available)",
)
args = parser.parse_args()

device = torch.device(args.device)

IMAGE_TOKENS = 256  # 2x2 latent patches: (4,32,32) -> 256 tokens of 16
DIM =1024
NUM_LAYERS=16
N_HEAD=16

# --- text encoder + vae (frozen, same as training/inference notebook) ---
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPTextModelWithProjection.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(device).eval()
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").eval().to(device)


def embed_text(text):
    inputs = processor(
        text=list(text),
        return_tensors="pt",
        padding="max_length",  # fixed 77 tokens, matching training
        truncation=True,
        max_length=77,
    ).to(device)
    with torch.inference_mode():
        outputs = clip_model(**inputs)
    return outputs.last_hidden_state  # B,77,512


def load_model(checkpoint):
    if not os.path.exists(checkpoint):
        print(f"Checkpoint {checkpoint} not found locally, downloading from {HF_REPO_ID}")
        checkpoint = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_CHECKPOINT)
    model = DiffusionModel(image_tokens=IMAGE_TOKENS, dim=DIM, num_layers=NUM_LAYERS, nhead=N_HEAD).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def generate(prompt, steps, guidance_scale, seed):
    if seed is not None and int(seed) >= 0:
        torch.manual_seed(int(seed))

    z = torch.randn([1, IMAGE_TOKENS, 16], device=device)
    prompt_emb = embed_text([prompt])
    null_prompt = embed_text([""])

    for i in range(int(steps)):
        t = torch.full((z.size(0), 1, 1), i / steps, device=device)
        v_null = MODEL(z, null_prompt, t)
        v_pred = MODEL(z, prompt_emb, t)
        v_latent = v_null + guidance_scale * (v_pred - v_null)
        z = v_latent * (1 / steps) + z

    # unpatchify: 256 tokens of (4ch x 2x2 patch) back to (4,32,32)
    z = z.reshape(1, 16, 16, 4, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(1, 4, 32, 32)
    z = z / vae.config.scaling_factor
    image = (vae.decode(z).sample / 2 + 0.5).clamp(0, 1)
    image = torch.nn.functional.interpolate(
        image, scale_factor=2, mode="bilinear", align_corners=False
    )
    image = image[0].permute(1, 2, 0).cpu().numpy()
    return image


def build_ui():
    with gr.Blocks(title="Text-to-Image (rectified flow)") as demo:
        gr.Markdown("# Text-to-Image\nRectified-flow diffusion in CLIP/VAE latent space.")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Prompt",
                    value="a red rose in a lush green garden. photorealistic",
                    lines=3,
                )
                steps = gr.Slider(1, 200, value=50, step=1, label="Steps")
                guidance = gr.Slider(0.0, 10.0, value=2.0, step=0.1, label="Guidance scale")
                seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
                btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                out = gr.Image(label="Output", type="numpy")
        btn.click(generate, [prompt, steps, guidance, seed], out)
    return demo


if __name__ == "__main__":
    MODEL = load_model(args.checkpoint)
    print(f"Loaded checkpoint {args.checkpoint} on {device}")

    build_ui().launch(server_name="0.0.0.0", share=True)
