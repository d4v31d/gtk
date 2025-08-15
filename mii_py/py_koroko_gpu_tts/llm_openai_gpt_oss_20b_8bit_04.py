
### llm_openai_gpt_oss_20b_8bit_04

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

#!/usr/bin/env python3
"""
Using Mxfp4Config for MX format quantization
"""

import torch
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    Mxfp4Config  # MX format quantization
)

def check_mx_support():
    """Check if MX quantization is supported"""
    try:
        from transformers import Mxfp4Config
        print("[INFO] Mxfp4Config is available")
        return True
    except ImportError:
        print("[ERROR] Mxfp4Config not available - update transformers")
        return False

def create_mxfp4_config(dequantize=True):
    """Create MX FP4 quantization configuration"""
    try:
        # Basic MX FP4 configuration
        mx_config = Mxfp4Config(
            dequantize=dequantize,  # Whether to dequantize during inference
            # Additional parameters you can set:
            # block_size=32,        # Block size for quantization
            # bits=4,               # Number of bits (usually 4 for FP4)
        )
        print(f"[INFO] Created Mxfp4Config with dequantize={dequantize}")
        return mx_config
    except Exception as e:
        print(f"[ERROR] Failed to create Mxfp4Config: {e}")
        return None

def load_model_with_mxfp4(model_id, dequantize=True):
    """Load model with MX FP4 quantization"""
    
    print(f"[INFO] Loading model: {model_id}")
    
    # Create MX FP4 config
    quantization_config = create_mxfp4_config(dequantize=dequantize)
    
    if quantization_config is None:
        print("[ERROR] Failed to create quantization config")
        return None, None
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("[INFO] Tokenizer loaded successfully")
        
        # Load model with MX FP4 quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        print("[INFO] Model loaded successfully with Mxfp4Config")
        return model, tokenizer
        
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None, None

def load_model_fallback_strategies(model_id):
    """Try multiple loading strategies including MX FP4"""
    
    strategies = [
        {
            "name": "MX FP4 with dequantize=True",
            "config": {
                "quantization_config": create_mxfp4_config(dequantize=True),
                "device_map": "auto",
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True
            }
        },
        {
            "name": "MX FP4 with dequantize=False", 
            "config": {
                "quantization_config": create_mxfp4_config(dequantize=False),
                "device_map": "auto",
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True
            }
        },
        {
            "name": "Standard FP16",
            "config": {
                "device_map": "auto",
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True
            }
        }
    ]
    
    # Load tokenizer first
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("[INFO] Tokenizer loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load tokenizer: {e}")
        return None, None
    
    # Try each strategy
    for strategy in strategies:
        try:
            print(f"[INFO] Trying: {strategy['name']}")
            
            # Skip if quantization config creation failed
            if (strategy['config'].get('quantization_config') is None and 
                'MX FP4' in strategy['name']):
                print(f"[SKIP] Quantization config is None")
                continue
                
            model = AutoModelForCausalLM.from_pretrained(model_id, **strategy['config'])
            print(f"[SUCCESS] Model loaded with: {strategy['name']}")
            return model, tokenizer
            
        except Exception as e:
            print(f"[FAILED] {strategy['name']}: {e}")
            continue
    
    print("[ERROR] All strategies failed")
    return None, None

def test_quantized_model(model, tokenizer, prompt="Hello, how are you?"):
    """Test the quantized model"""
    if model is None or tokenizer is None:
        print("[ERROR] Cannot test: model or tokenizer is None")
        return
    
    print(f"[INFO] Testing model with prompt: '{prompt}'")
    print(f"[INFO] Model device: {next(model.parameters()).device}")
    print(f"[INFO] Model dtype: {next(model.parameters()).dtype}")
    
    try:
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"[INFO] Response: {response}")
        
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")

def compare_quantization_methods(model_id):
    """Compare different quantization approaches"""
    print(f"\n[INFO] Comparing quantization methods for {model_id}")
    
    methods = [
        ("MX FP4 (dequantize=True)", lambda: create_mxfp4_config(dequantize=True)),
        ("MX FP4 (dequantize=False)", lambda: create_mxfp4_config(dequantize=False)),
    ]
    
    for name, config_func in methods:
        print(f"\n--- Testing {name} ---")
        config = config_func()
        if config:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=config,
                    device_map="cpu",  # Use CPU for memory comparison
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
                
                # Check memory usage
                param_size = sum(p.numel() * p.element_size() for p in model.parameters())
                print(f"[INFO] Model size: {param_size / 1024**2:.2f} MB")
                print(f"[INFO] Success with {name}")
                
                del model  # Free memory
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print(f"[ERROR] Failed with {name}: {e}")

def main():
    """Main function demonstrating Mxfp4Config usage"""
    
    # Check if MX quantization is supported
    if not check_mx_support():
        print("[ERROR] Please update transformers to use Mxfp4Config")
        print("pip install --upgrade transformers")
        return
    
    # Model configuration
    MODEL_ID = "microsoft/DialoGPT-medium"  # Change to your model
    
    print(f"\n=== Loading {MODEL_ID} with MX FP4 Quantization ===")
    
    # Method 1: Direct usage
    print("\n--- Method 1: Direct Mxfp4Config ---")
    model, tokenizer = load_model_with_mxfp4(MODEL_ID, dequantize=True)
    
    if model and tokenizer:
        test_quantized_model(model, tokenizer)
    else:
        print("\n--- Method 2: Fallback Strategies ---")
        model, tokenizer = load_model_fallback_strategies(MODEL_ID)
        if model and tokenizer:
            test_quantized_model(model, tokenizer)
    
    # Method 3: Compare different settings
    print("\n--- Method 3: Comparison ---")
    compare_quantization_methods(MODEL_ID)

# Example configurations for different use cases
def example_configs():
    """Show different Mxfp4Config examples"""
    
    examples = {
        "Standard MX FP4": Mxfp4Config(dequantize=True),
        "Memory Optimized": Mxfp4Config(dequantize=False),
        # Add more configurations as needed
    }
    
    for name, config in examples.items():
        print(f"{name}: {config}")

if __name__ == "__main__":
    main()
    
    print("\n--- Example Configurations ---")
    example_configs()
    