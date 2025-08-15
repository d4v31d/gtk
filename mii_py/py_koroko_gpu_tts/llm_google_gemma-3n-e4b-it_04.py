
### llm_google_gemma-3n-e4b-it_04

from transformers import pipeline
import torch

# pipe = pipeline(
#     "image-text-to-text",
#     model="google/gemma-3n-e4b-it",
#     device="cuda",
#     torch_dtype=torch.bfloat16,
# )

### working til here...

# messages = [
#     {
#         "role": "system",
#         "content": [{"type": "text", "text": "You are a helpful assistant."}]
#     },
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
#             {"type": "text", "text": "What animal is on the candy?"}
#         ]
#     }
# ]

# output = pipe(
#     text=messages
#     ,max_new_tokens=200
# )

# print(output[0]["generated_text"][-1]["content"])

# Use regular Gemma 2 for text-only tasks
pipe = pipeline(
    "text-generation",
    model="google/gemma-2-2b-it",  # or gemma-2-9b-it
    device="cpu",
    #device_map="auto"
)

# Format messages properly for chat
messages = [
    {"role": "user", "content": "Make a json file with top 30 capital in the world and their middle temperature in celsius in august."},
]

output = pipe(
    messages,
    max_new_tokens=200
)

print(output[0]["generated_text"])
