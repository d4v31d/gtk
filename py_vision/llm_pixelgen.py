
### llm_pixelgen 
### make an image using the PixelGen model from diffusers
import torch
from diffusers import DiffusionPipeline
import os
import time

ts = time.strftime("%Y%m%d%H%M%S")

time_begin = time.time()

model = "OEvortex/PixelGen"

pipe = DiffusionPipeline.from_pretrained(
    model,
    local_files_only=True,
    force_download=False,
    torch_dtype=torch.float16
    # torch_dtype="auto"
)


# pipe = DiffusionPipeline.from_pretrained(
#     "OEvortex/PixelGen",
#     # ignore_mismatched_sizes=True
#     # revision="main",  # or try "fp16", "fp32", etc.
#     # torch_dtype=torch.float16  # if using GPU
#     torch_dtype="auto"
# )

prompt = "a businessman in proper zebra patterned clothes in a concrete jungle, landscape format, wide screen format, cold color palette, muted colors, detailed, max 2k resolution, cinematic lighting, highly detailed, intricate, sharp focus, depth of field, ultra realistic, hyper realistic, trending on artstation, by Greg Rutkowski and Artgerm and Alphonse Mucha and Studio Ghibli"
image = pipe(prompt).images[0]

image.save(f"mii_img_{ts}.png")
print("Image saved as astronaut_in_jungle.png")
#image.show()

time_end = time.time()

print("-" * 70)
print("=== PixelGen Image Generation Report ===")
print(f"Prompt: {prompt}")
print(f"Model: OEvortex/PixelGen")
print(f"Begin: {time_begin}")
print(f"End: {time_end}")
print(f"Image saved as: mii_img_{ts}.png")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"Image size: {image.size}")
print(f"Image format: {image.format}")
print(f"Time taken: {time_end - time_begin:.2f} seconds")

"""

pip install uv
uv pip install pip --upgrade pip wheel setuptools

uv pip install diffusers
uv pip install transformers
uv pip install torch
uv pip install accelerate
uv pip install --upgrade diffusers transformers torch huggingface_hub

# Update PyTorch for RTX 4070 support
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Update diffusion libraries
uv pip install --upgrade diffusers transformers accelerate xformers

# Optional: Install for better performance
uv pip install triton

"""

