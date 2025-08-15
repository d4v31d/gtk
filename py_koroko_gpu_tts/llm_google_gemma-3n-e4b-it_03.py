
### llm_google_gemma-3n-e4b-it_03

### hf auth login --token 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
# The token has not been saved to the git credentials helper. Pass `add_to_git_credential=True` in this function directly or `--add-to-git-credential` if using via `hf`CLI if you want to set the git credential as well.
# Token is valid (permission: read).
# The token `galningen` has been saved to C:\Users\david\.cache\huggingface\stored_tokens
# Your token has been saved to C:\Users\david\.cache\huggingface\token
# Login successful.
# The current active token is: `galningen`


# Load model directly
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("google/gemma-3n-E4B-it")
model = AutoModelForImageTextToText.from_pretrained("google/gemma-3n-E4B-it")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "candy.jpg"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    },
]
inputs = processor.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))


