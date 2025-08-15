
### llm_openai_gpt_oss_20b_8bit_02

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

error again, trying this
uv pip install --upgrade "transformers>=4.40.0" accelerate bitsandbytes

check versions
pip show transformers bitsandbytes
Name: transformers  Version: 4.55.0
Name: bitsandbytes  Version: 0.46.1

check cached versions
python -c "import transformers; print(transformers.__version__, transformers.__file__)"

"""

import torch
import transformers
from packaging import version

MODEL_ID = "openai/gpt-oss-20b"

def main():
    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    # Detect transformers version
    tf_version = version.parse(transformers.__version__)
    print(f"[INFO] Using transformers {tf_version}")

    if tf_version >= version.parse("4.38.0"):
        # New API path
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=True
        )
        print("[INFO] Loading model with BitsAndBytesConfig (new API)")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
    else:
        # Legacy path
        print("[INFO] Loading model with load_in_8bit=True (legacy API)")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            load_in_8bit=True,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

    # Run inference without pipeline (avoids torchvision)
    prompt = "List 5 benefits of running an open-source LLM locally:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False, temperature=0.0)

    print("\n=== Model Output ===\n")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
