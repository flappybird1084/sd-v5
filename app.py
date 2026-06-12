import argparse
import base64
import io
import os

import gradio as gr
import torch
from diffusers import AutoencoderKL
from huggingface_hub import hf_hub_download
from PIL import Image
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
DIM = 1024
NUM_LAYERS = 16
N_HEAD = 16

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


def decode_latent(z):
    # unpatchify: 256 tokens of (4ch x 2x2 patch) back to (4,32,32)
    z = z.reshape(1, 16, 16, 4, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(1, 4, 32, 32)
    z = z / vae.config.scaling_factor
    image = (vae.decode(z).sample / 2 + 0.5).clamp(0, 1)
    return image[0].permute(1, 2, 0).cpu().numpy()


def to_jpeg_b64(image):
    pil = Image.fromarray((image * 255).astype("uint8"))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


IMG_STYLE = "width:100%;max-width:512px;border-radius:8px;image-rendering:auto;"


def progress_html(b64, step, total):
    return (
        '<div style="text-align:center;">'
        f'<img src="data:image/jpeg;base64,{b64}" style="{IMG_STYLE}"/>'
        f'<p style="font-family:monospace;">step {step} / {total}&hellip;</p>'
        "</div>"
    )


def scrubber_html(frames, labels, total):
    n = len(frames)
    imgs = "".join(
        f'<img class="rf-frame" src="data:image/jpeg;base64,{b64}" '
        f'style="{IMG_STYLE}display:{"inline" if i == n - 1 else "none"};"/>'
        for i, b64 in enumerate(frames)
    )
    # inline handlers only: gr.HTML inserts via innerHTML, so <script> tags would not run
    oninput = (
        "const c=this.closest('.rf-scrub');"
        "c.querySelectorAll('.rf-frame').forEach((im,i)=>im.style.display=i==this.value?'inline':'none');"
        "c.querySelector('.rf-step').textContent="
        f"'step '+c.dataset.labels.split(',')[this.value]+' / {total}';"
    )
    slider = (
        f'<input type="range" min="0" max="{n - 1}" value="{n - 1}" step="1" '
        f'style="width:100%;max-width:512px;display:block;margin:8px auto;" oninput="{oninput}"/>'
        if n > 1
        else ""  # single frame: nothing to scrub
    )
    return (
        f'<div class="rf-scrub" data-labels="{",".join(str(s) for s in labels)}" style="text-align:center;">'
        f"{imgs}"
        f"{slider}"
        f'<p class="rf-step" style="font-family:monospace;">step {labels[-1]} / {total}</p>'
        "</div>"
    )


def generate(prompt, steps, guidance_scale, seed, preview_every, progress=gr.Progress()):
    if seed is not None and int(seed) >= 0:
        torch.manual_seed(int(seed))

    steps = max(1, int(steps))
    # clamp: 0/negative would crash the modulo below, > steps means final frame only
    preview_every = max(1, min(int(preview_every), steps))

    with torch.no_grad():
        progress(0, desc="encoding prompt")
        z = torch.randn([1, IMAGE_TOKENS, 16], device=device)
        prompt_emb = embed_text([prompt])
        null_prompt = embed_text([""])

        frames, labels = [], []
        for i in range(steps):
            t = torch.full((z.size(0), 1, 1), i / steps, device=device)
            v_null = MODEL(z, null_prompt, t)
            v_pred = MODEL(z, prompt_emb, t)
            v_latent = v_null + guidance_scale * (v_pred - v_null)
            z = v_latent * (1 / steps) + z

            step = i + 1
            progress(step / steps, desc=f"step {step}/{steps}")
            if step % preview_every == 0 or step == steps:
                b64 = to_jpeg_b64(decode_latent(z))
                frames.append(b64)
                labels.append(step)
                if step < steps:
                    yield progress_html(b64, step, steps)

    yield scrubber_html(frames, labels, steps)


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
                preview_every = gr.Slider(
                    1, 200, value=2, step=1,
                    label="Preview every N steps (each preview adds a VAE decode)",
                )
                btn = gr.Button("Generate", variant="primary")
                status = gr.HTML()  # empty; hosts the progress bar/timer during generation
            with gr.Column():
                out = gr.HTML(label="Output")
        btn.click(
            generate,
            [prompt, steps, guidance, seed, preview_every],
            out,
            show_progress="full",
            show_progress_on=status,
        )
    return demo


if __name__ == "__main__":
    MODEL = load_model(args.checkpoint)
    print(f"Loaded checkpoint {args.checkpoint} on {device}")

    build_ui().launch(server_name="0.0.0.0", share=True)
