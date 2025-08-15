
### llm_openai_gpt_oss_20b_8bit_03

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

# Check your current versions
pip show transformers bitsandbytes torch

    Name: bitsandbytes
    Version: 0.46.1
    Location: C:\tmp\mii_py\py_koroko_gpu_tts\.venv\Lib\site-packages
    Requires: numpy, torch
    Required-by:
    ---
    Name: torch
    Version: 2.8.0+cu128
    Location: C:\tmp\mii_py\py_koroko_gpu_tts\.venv\Lib\site-packages
    Requires: filelock, fsspec, jinja2, networkx, setuptools, sympy, typing-extensions
    Required-by: accelerate, bitsandbytes, curated-transformers, kokoro, spacy-curated-transformers, timm, torchaudio, torchvision
    ---
    Name: transformers
    Version: 4.55.0
    Location: C:\tmp\mii_py\py_koroko_gpu_tts\.venv\Lib\site-packages
    Requires: filelock, huggingface-hub, numpy, packaging, pyyaml, regex, requests, safetensors, tokenizers, tqdm
    Required-by: kokoro

# Check CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

    CUDA: True

"""

#!/usr/bin/env python3
"""
Fixed LLM loading code with proper BitsAndBytesConfig setup
"""

import os
import torch
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline
)

def check_versions():
    """Check package versions for debugging"""
    print(f"[INFO] PyTorch version: {torch.__version__}")
    print(f"[INFO] Transformers version: {transformers.__version__}")
    try:
        import bitsandbytes as bnb
        print(f"[INFO] BitsAndBytes version: {bnb.__version__}")
    except ImportError:
        print("[WARNING] BitsAndBytes not installed")
    print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] CUDA device: {torch.cuda.get_device_name()}")

def create_bnb_config(load_in_8bit=True, load_in_4bit=False):
    """Create BitsAndBytes configuration"""
    if load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif load_in_8bit:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
            bnb_8bit_quant_type="nf8",
            bnb_8bit_use_double_quant=True,
        )
    else:
        return None

def load_model_and_tokenizer(model_id, use_quantization=True, load_in_4bit=False):
    """Load model and tokenizer with error handling"""
    
    print(f"[INFO] Loading model: {model_id}")
    
    # Create quantization config
    if use_quantization:
        try:
            quantization_config = create_bnb_config(
                load_in_8bit=not load_in_4bit, 
                load_in_4bit=load_in_4bit
            )
            print(f"[INFO] Using quantization: {'4-bit' if load_in_4bit else '8-bit'}")
        except Exception as e:
            print(f"[WARNING] Failed to create quantization config: {e}")
            print("[INFO] Falling back to no quantization")
            quantization_config = None
    else:
        quantization_config = None
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("[INFO] Tokenizer loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load tokenizer: {e}")
        return None, None
    
    # Load model with different fallback strategies
    model = None
    strategies = [
        # Strategy 1: With quantization
        {
            "quantization_config": quantization_config,
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True
        },
        # Strategy 2: Without quantization but with device mapping
        {
            "device_map": "auto", 
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True
        },
        # Strategy 3: Basic loading
        {
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True
        }
    ]
    
    for i, strategy in enumerate(strategies, 1):
        if not use_quantization and i == 1:
            continue  # Skip quantization strategy if not wanted
            
        try:
            print(f"[INFO] Trying loading strategy {i}...")
            model = AutoModelForCausalLM.from_pretrained(model_id, **strategy)
            print(f"[INFO] Model loaded successfully with strategy {i}")
            break
        except Exception as e:
            print(f"[WARNING] Strategy {i} failed: {e}")
            continue
    
    if model is None:
        print("[ERROR] All loading strategies failed")
        return None, None
    
    return model, tokenizer

def test_model(model, tokenizer, prompt="Hello, how are you?"):
    """Test the loaded model"""
    if model is None or tokenizer is None:
        print("[ERROR] Cannot test: model or tokenizer is None")
        return
    
    print(f"[INFO] Testing model with prompt: '{prompt}'")
    
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"[INFO] Generated response: {response}")
        
    except Exception as e:
        print(f"[ERROR] Failed to generate response: {e}")

def main():
    """Main function"""
    # Check system info
    check_versions()
    
    # Model configuration - CHANGE THIS TO YOUR MODEL
    MODEL_ID = "microsoft/DialoGPT-medium"  # Example model, replace with yours
    # MODEL_ID = "huggingface/CodeBERTa-small-v1"  # Another example
    # MODEL_ID = "your-model-name-here"
    
    USE_QUANTIZATION = True  # Set to False if you have issues
    USE_4BIT = False  # Set to True for 4-bit instead of 8-bit
    
    print(f"\n[INFO] Configuration:")
    print(f"  Model: {MODEL_ID}")
    print(f"  Quantization: {USE_QUANTIZATION}")
    print(f"  4-bit: {USE_4BIT}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        MODEL_ID, 
        use_quantization=USE_QUANTIZATION,
        load_in_4bit=USE_4BIT
    )
    
    if model is not None and tokenizer is not None:
        print(f"[INFO] Model loaded successfully!")
        print(f"[INFO] Model device: {model.device if hasattr(model, 'device') else 'Unknown'}")
        print(f"[INFO] Model dtype: {model.dtype if hasattr(model, 'dtype') else 'Unknown'}")
        
        # Test the model
        test_model(model, tokenizer, "Hello, how are you today?")
    else:
        print("[ERROR] Failed to load model")

if __name__ == "__main__":
    main()

