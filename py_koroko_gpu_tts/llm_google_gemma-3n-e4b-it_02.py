
### llm_google_gemma-3n-e4b-it_02

from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from PIL import Image
import requests
import torch
import time

def load_image(path):
    with open(path, "rb") as f:
        return f.read()
    

ts_begin = time.strftime("%Y%m%d%H%M%S")
print(f"ts_begin: {ts_begin}")
print()

model_id = "google/gemma-3n-e4b-it"
print(f"model_id: {model_id}")
print()

model = Gemma3nForConditionalGeneration.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16,).eval()

processor = AutoProcessor.from_pretrained(model_id)

image_bytes = load_image("candy.jpg")

# messages = [
#     {
#         "role": "system",
#         "content": "What animal is on the candy?",
#         "images": [image_bytes],
#     },
# ]


messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
            {"type": "text", "text": "Describe this image in detail."}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]

decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)

# **Overall Impression:** The image is a close-up shot of a vibrant garden scene,
# focusing on a cluster of pink cosmos flowers and a busy bumblebee.
# It has a slightly soft, natural feel, likely captured in daylight.


ts_end = time.strftime("%Y%m%d%H%M%S")
print()
print(f"ts_end: {ts_end}")
print()
print()














# from transformers import pipeline
# import torch
# import time

# ts_begin = time.strftime("%Y%m%d%H%M%S")
# print(f"ts_begin: {ts_begin}")
# print()

# model_id="google/gemma-3n-e4b-it"
# print(f"model_id: {model_id}")
# print()

# pipe = pipeline(
#     "image-text-to-text",
#     model=model_id,
#     device="cuda",
#     torch_dtype=torch.bfloat16,
# )

# messages = [
#     {
#         "role": "system",
#         "content": [{"type": "text", "text": "You are a helpful assistant."}]
#     },
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "url": "candy.jpg"},
#             {"type": "text", "text": "What animal is on the candy?"}
#         ]
#     }
# ]

# output = pipe(
#     text=messages
#     ,max_new_tokens=200
# )

# print(output[0]["generated_text"][-1]["content"])

# # Okay, let's take a look!
# # Based on the image, the animal on the candy is a **turtle**.
# # You can see the shell shape and the head and legs.

# ts_end = time.strftime("%Y%m%d%H%M%S")
# print(f"ts_end: {ts_end}")
# print()

