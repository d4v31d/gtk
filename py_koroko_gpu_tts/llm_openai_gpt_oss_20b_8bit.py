
### llm_openai_gpt_oss_20b_8bit

"""

## alt 1
# basic python deps (adjust CUDA / torch version for your machine)
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers accelerate

# optionally for quantized/8-bit support:
pip install bitsandbytes

# If you want the MXFP4/triton path mentioned in docs:
pip install git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels


## alt 2
# install vllm (follow vLLM docs)
pip install vllm

# run vLLM server with the model (example; exact CLI args from vLLM docs)
vllm --model openai/gpt-oss-20b --port 8000


## alt win 11 rtx 4070 8gb
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install transformers accelerate bitsandbytes

error because of old version of transformers, use this to upgrade:
uv pip install --upgrade transformers accelerate bitsandbytes
pip show transformers | findstr Version

error again with torchvision version mismatch

First, check your PyTorch version:
pip show torch | findstr Version  ## Version: 2.8.0

Then uninstall the mismatched TorchVision:
uv pip uninstall torchvision -y

Reinstall the matching version:
# For PyTorch 2.4.0 (CUDA 12.1)  ## i have cuda 12.8
uv pip install torchvision --index-url https://download.pytorch.org/whl/cu128

AttributeError: 'BitsAndBytesConfig' object has no attribute 'get_loading_attributes'
use Mxfp4Config

"""

# run_gpt_oss_20b_bitsandbytes.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from transformers import Mxfp4Config, AutoModelForCausalLM
from transformers import GptOssForCausalLM

quantization_config = Mxfp4Config(dequantize=False)

MODEL_ID = "openai/gpt-oss-20b"

def main():
    # Configure BitsAndBytes for 8-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,        # 8-bit weights
        llm_int8_threshold=6.0,   # outlier threshold for slower layers
        llm_int8_has_fp16_weight=True
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        #device_map="auto",               # offload if needed
        torch_dtype="auto",
        # quantization_config=bnb_config   # <-- new way
        # torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )

    prompt = "List 5 benefits of running an open-source LLM locally:"
    result = generator(prompt, max_new_tokens=150, do_sample=False, temperature=0.0)

    print(result[0]["generated_text"])

if __name__ == "__main__":
    main()





# # run_gpt_oss_20b_8bit.py
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# MODEL_ID = "openai/gpt-oss-20b"

# def main():
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_ID,
#         device_map="auto",          # auto-assign layers to GPU/CPU
#         load_in_8bit=True,          # 8-bit quantization
#         torch_dtype=torch.float16,  # half precision where possible
#         low_cpu_mem_usage=True
#     )

#     gen = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         device=0 if torch.cuda.is_available() else -1
#     )

#     prompt = "Explain the benefits of running a local large language model."
#     out = gen(prompt, max_new_tokens=200, do_sample=False, temperature=0.0)

#     print(out[0]["generated_text"])

# if __name__ == "__main__":
#     main()

# The `load_in_4bit` and `load_in_8bit` arguments are deprecated and will be removed in the future versions. Please, pass a `BitsAndBytesConfig` object in `quantization_config` argument instead.




