
### llm_google_gemma-3n-e4b-it_01

from transformers import pipeline
import torch
import time

ts_begin = time.strftime("%Y%m%d%H%M%S")
print(f"ts_begin: {ts_begin}")
print()

model_id="google/gemma-3n-e4b-it"
print(f"model_id: {model_id}")
print()

# hf-auth-login
pipe = pipeline(
    "image-text-to-text",
    model=model_id,
    device="cuda",
    torch_dtype=torch.bfloat16,
)

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "candy.jpg"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    }
]

output = pipe(
    text=messages
    ,max_new_tokens=200
)

print(output[0]["generated_text"][-1]["content"])

# Okay, let's take a look!
# Based on the image, the animal on the candy is a **turtle**.
# You can see the shell shape and the head and legs.

ts_end = time.strftime("%Y%m%d%H%M%S")
print(f"ts_end: {ts_end}")
print()

