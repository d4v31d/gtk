
### llm_qwen3-8b_01

from transformers import AutoModelForCausalLM, AutoTokenizer
import time


ts_begin = time.strftime("%Y%m%d%H%M%S")
print(f"ts_begin: {ts_begin}")
print()

model_name = "Qwen/Qwen3-8B"
print(f"model_name: {model_name}")
print()

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input
prompt = "generate python code for processing a folder with files. After they are done, moves them to processed_folder."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=200 ##32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)


ts_end = time.strftime("%Y%m%d%H%M%S")
print()
print(f"ts_end: {ts_end}")
print()
print()



# `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`
