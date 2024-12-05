import os
import time
import uuid

import gradio as gr
import numpy as np
import torch
from einops import rearrange
from PIL import ExifTags, Image
from transformers import pipeline

from flux.sampling import unpack, prepare_minimal, sample_minimal
from flux.util import embed_watermark, load_ae, load_minimal_model

NSFW_THRESHOLD = 0.85

def get_models(device: torch.device, offload: bool):
    model = load_minimal_model(device="cpu" if offload else device)
    ae = load_ae("flux-dev", device="cpu" if offload else device)
    nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)
    return model, ae, nsfw_classifier

class MinimalFluxGenerator:
    def __init__(self, device: str, offload: bool):
        self.device = torch.device(device)
        self.offload = offload
        self.model, self.ae, self.nsfw_classifier = get_models(
            device=self.device,
            offload=self.offload,
        )

    @torch.inference_mode()
    def generate_image(
        self,
        width: int,
        height: int,
        num_steps: int,
        seed: int,
        add_sampling_metadata: bool = True,
    ):
        seed = int(seed)
        if seed == -1:
            seed = None

        if seed is None:
            seed = torch.Generator(device="cpu").seed()
        
        print(f"Generating with seed {seed}")
        t0 = time.perf_counter()

        # Use the minimal sampling function
        x = sample_minimal(
            model=self.model,
            height=height,
            width=width,
            num_steps=num_steps,
            seed=seed,
            device=self.device
        )

        # Decode latents to pixel space
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        print(f"Done in {t1 - t0:.1f}s.")
        
        # Process and save image
        x = x.clamp(-1, 1)
        x = embed_watermark(x.float())
        x = rearrange(x[0], "c h w -> h w c")

        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        nsfw_score = [x["score"] for x in self.nsfw_classifier(img) if x["label"] == "nsfw"][0]

        if nsfw_score < NSFW_THRESHOLD:
            filename = f"output/gradio_minimal/{uuid.uuid4()}.jpg"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            exif_data = Image.Exif()
            exif_data[ExifTags.Base.Software] = "AI generated;minimal-flux"
            exif_data[ExifTags.Base.Make] = "Black Forest Labs"
            exif_data[ExifTags.Base.Model] = "flux-minimal"

            img.save(filename, format="jpeg", exif=exif_data, quality=95, subsampling=0)

            return img, str(seed), filename, None
        else:
            return None, str(seed), None, "Your generated image may contain NSFW content."

def create_demo(
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    offload: bool = False
):
    generator = MinimalFluxGenerator(device, offload)

    with gr.Blocks() as demo:
        gr.Markdown("# Minimal FLUX Demo - Unconditional Image Generation")

        with gr.Row():
            with gr.Column():
                with gr.Accordion("Generation Parameters", open=True):
                    width = gr.Slider(128, 8192, 1024, step=16, label="Width")
                    height = gr.Slider(128, 8192, 1024, step=16, label="Height")
                    num_steps = gr.Slider(1, 50, 50, step=1, label="Number of steps")
                    seed = gr.Textbox(-1, label="Seed (-1 for random)")
                    add_sampling_metadata = gr.Checkbox(
                        label="Add sampling parameters to metadata?",
                        value=True
                    )

                generate_btn = gr.Button("Generate")

            with gr.Column():
                output_image = gr.Image(label="Generated Image")
                seed_output = gr.Number(label="Used Seed")
                warning_text = gr.Textbox(label="Warning", visible=False)
                download_btn = gr.File(label="Download full-resolution")

        generate_btn.click(
            fn=generator.generate_image,
            inputs=[
                width,
                height,
                num_steps,
                seed,
                add_sampling_metadata,
            ],
            outputs=[output_image, seed_output, download_btn, warning_text],
        )

    return demo

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Minimal Flux Demo")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Offload model to CPU when not in use"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public link to your demo"
    )
    args = parser.parse_args()

    demo = create_demo(args.device, args.offload)
    demo.launch(share=args.share)