
### llm_openai_gpt_oss_20b_8bit_05

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

    

Solution 1: Install Triton (Recommended)
bash# Install Triton 3.4.0 or newer
pip install triton>=3.4.0

# Install triton_kernels (this might be the missing piece)
pip install triton-kernels

# OR try:
pip install git+https://github.com/openai/triton.git

Solution 2: If Triton installation fails
Sometimes Triton can be tricky to install. Try these alternatives:
bash# Try with specific CUDA version
pip install triton>=3.4.0 --extra-index-url https://download.pytorch.org/whl/cu118


pip install --upgrade pip setuptools wheel

nvcc --version

    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2025 NVIDIA Corporation
    Built on Wed_Jul_16_20:06:48_Pacific_Daylight_Time_2025
    Cuda compilation tools, release 13.0, V13.0.48
    Build cuda_13.0.r13.0/compiler.36260728_0


"""

#!/usr/bin/env python3
"""
Fixed LLM loading code with proper BitsAndBytesConfig setup
"""

#!/usr/bin/env python3
"""
Using Mxfp4Config for MX format quantization
"""

#!/usr/bin/env python3
"""
Handling MXFP4 quantization with Triton dependency issues
"""

import torch
import transformers
import warnings
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    Mxfp4Config
)

def check_triton_support():
    """Check Triton installation and version"""
    try:
        import triton
        print(f"[INFO] Triton version: {triton.__version__}")
        
        # Check if version is sufficient
        version_parts = triton.__version__.split('.')
        major, minor = int(version_parts[0]), int(version_parts[1])
        
        if major > 3 or (major == 3 and minor >= 4):
            print("[INFO] Triton version is sufficient (>=3.4.0)")
            triton_ok = True
        else:
            print(f"[WARNING] Triton version {triton.__version__} < 3.4.0")
            triton_ok = False
            
    except ImportError:
        print("[WARNING] Triton not installed")
        triton_ok = False
    
    # Check triton_kernels
    try:
        import triton_kernels
        print("[INFO] triton_kernels is available")
        kernels_ok = True
    except ImportError:
        print("[WARNING] triton_kernels not installed")
        kernels_ok = False
    
    return triton_ok and kernels_ok

def install_triton_instructions():
    """Print installation instructions for Triton"""
    print("\n=== Triton Installation Instructions ===")
    print("Try these commands:")
    print("1. pip install triton>=3.4.0")
    print("2. pip install triton-kernels")
    print("3. If that fails, try:")
    print("   pip install triton>=3.4.0 --extra-index-url https://download.pytorch.org/whl/cu118")
    print("4. Or with conda:")
    print("   conda install triton>=3.4.0 -c conda-forge")
    print("==========================================\n")

def load_model_with_mxfp4_fallback(model_id, handle_triton_warning=True):
    """Load model with MXFP4, handling Triton warnings gracefully"""
    
    print(f"[INFO] Loading {model_id} with MXFP4 quantization")
    
    # Check Triton support
    triton_supported = check_triton_support()
    
    if not triton_supported:
        print("[INFO] Triton requirements not met - model will fall back to bf16")
        if not handle_triton_warning:
            install_triton_instructions()
            return None, None
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create MXFP4 config
        quantization_config = Mxfp4Config(dequantize=True)
        
        # Suppress the Triton warning if desired
        if handle_triton_warning:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*MXFP4 quantization requires triton.*")
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,  # Use bfloat16 as fallback
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        
        print("[INFO] Model loaded successfully (with or without Triton)")
        print(f"[INFO] Actual model dtype: {next(model.parameters()).dtype}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None, None

def alternative_quantization_strategies(model_id):
    """Try different quantization approaches if MXFP4 doesn't work well"""
    
    strategies = [
        {
            "name": "MXFP4 (with Triton fallback)",
            "loader": lambda: load_model_with_mxfp4_fallback(model_id, handle_triton_warning=True)
        },
        {
            "name": "BitsAndBytes 8-bit",
            "loader": lambda: load_with_bnb_8bit(model_id)
        },
        {
            "name": "BitsAndBytes 4-bit", 
            "loader": lambda: load_with_bnb_4bit(model_id)
        },
        {
            "name": "Standard bfloat16",
            "loader": lambda: load_standard_bf16(model_id)
        }
    ]
    
    for strategy in strategies:
        print(f"\n--- Trying: {strategy['name']} ---")
        try:
            model, tokenizer = strategy['loader']()
            if model is not None:
                print(f"[SUCCESS] {strategy['name']} worked!")
                return model, tokenizer
            else:
                print(f"[FAILED] {strategy['name']} returned None")
        except Exception as e:
            print(f"[FAILED] {strategy['name']}: {e}")
    
    return None, None

def load_with_bnb_8bit(model_id):
    """Fallback to BitsAndBytes 8-bit"""
    try:
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        return model, tokenizer
        
    except Exception as e:
        print(f"BitsAndBytes 8-bit failed: {e}")
        return None, None

def load_with_bnb_4bit(model_id):
    """Fallback to BitsAndBytes 4-bit"""
    try:
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        return model, tokenizer
        
    except Exception as e:
        print(f"BitsAndBytes 4-bit failed: {e}")
        return None, None

def load_standard_bf16(model_id):
    """Fallback to standard bfloat16"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Standard bfloat16 failed: {e}")
        return None, None

def test_model_performance(model, tokenizer, prompt="Hello, how are you?"):
    """Test model and measure performance"""
    if model is None or tokenizer is None:
        print("[ERROR] Cannot test: model or tokenizer is None")
        return
    
    print(f"[INFO] Testing with prompt: '{prompt}'")
    print(f"[INFO] Model dtype: {next(model.parameters()).dtype}")
    print(f"[INFO] Model device: {next(model.parameters()).device}")
    
    # Check memory usage
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()
    
    try:
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate with timing
        import time
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        end_time = time.time()
        
        # Decode and report
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"[INFO] Generated: {response}")
        print(f"[INFO] Generation time: {end_time - start_time:.2f}s")
        
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated()
            print(f"[INFO] Memory used: {(peak_memory - initial_memory) / 1024**2:.2f} MB")
        
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")

def main():
    """Main function with comprehensive MXFP4 handling"""
    
    MODEL_ID = "microsoft/DialoGPT-medium"  # Change to your model
    
    print("=== MXFP4 Quantization with Triton Fallback ===\n")
    
    # Method 1: Direct MXFP4 with graceful fallback
    print("--- Method 1: MXFP4 with fallback handling ---")
    model, tokenizer = load_model_with_mxfp4_fallback(MODEL_ID)
    
    if model and tokenizer:
        test_model_performance(model, tokenizer)
        print("[SUCCESS] MXFP4 method completed")
    else:
        print("\n--- Method 2: Alternative quantization strategies ---")
        model, tokenizer = alternative_quantization_strategies(MODEL_ID)
        
        if model and tokenizer:
            test_model_performance(model, tokenizer)
        else:
            print("[ERROR] All methods failed")

if __name__ == "__main__":
    main()

