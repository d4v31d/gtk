
### llm_openai_gpt-oss-20b
### llm is TO BIG damnit, need more gpu ram...

# uv pip install -U transformers kernels torch 

from transformers import pipeline
import torch
import time

ts_begin = time.strftime("%Y%m%d%H%M%S")
print()
print("-" * 70)
print(f"ts_begin: {ts_begin}")


model_id = "openai/gpt-oss-20b"
print(f"model_id: {model_id}")
print()

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
]

outputs = pipe(
    messages,
    max_new_tokens=256,
)

print(outputs[0]["generated_text"][-1])
print()

ts_end = time.strftime("%Y%m%d%H%M%S")
print(f"ts_end: {ts_end}")
print()
print()
