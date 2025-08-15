
## llm_pixelgen_diag

# Diagnostic script for PixelGen loading issues
import torch
import os
from huggingface_hub import hf_hub_download, list_repo_files
from diffusers import DiffusionPipeline
import json

def diagnose_pixelgen():
    model_id = "OEvortex/PixelGen"
    
    print("=== PixelGen Diagnostic Report ===\n")
    
    # 1. Check what files are in the repository
    print("1. Repository Contents:")
    try:
        files = list_repo_files(model_id)
        for file in sorted(files):
            print(f"   {file}")
    except Exception as e:
        print(f"   Error listing files: {e}")
    
    print()
    
    # 2. Check model_index.json
    print("2. Model Configuration:")
    try:
        model_index_path = hf_hub_download(model_id, "model_index.json")
        with open(model_index_path, 'r') as f:
            model_index = json.load(f)
        print("   model_index.json contents:")
        for key, value in model_index.items():
            print(f"   {key}: {value}")
    except Exception as e:
        print(f"   Error reading model_index.json: {e}")
    
    print()
    
    # 3. Check text encoder config
    print("3. Text Encoder Configuration:")
    try:
        config_path = hf_hub_download(model_id, "text_encoder/config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"   Hidden size: {config.get('hidden_size', 'Not found')}")
        print(f"   Vocab size: {config.get('vocab_size', 'Not found')}")
        print(f"   Max position embeddings: {config.get('max_position_embeddings', 'Not found')}")
    except Exception as e:
        print(f"   Error reading text_encoder config: {e}")
    
    print()
    
    # 4. Try loading individual components
    print("4. Component Loading Test:")
    components_to_test = ['text_encoder', 'tokenizer', 'unet', 'vae', 'scheduler']
    
    for component in components_to_test:
        try:
            if component == 'text_encoder':
                from transformers import CLIPTextModel
                model = CLIPTextModel.from_pretrained(f"{model_id}", subfolder=component)
                print(f"   ✓ {component}: Loaded successfully")
            elif component == 'tokenizer':
                from transformers import CLIPTokenizer
                tokenizer = CLIPTokenizer.from_pretrained(f"{model_id}", subfolder=component)
                print(f"   ✓ {component}: Loaded successfully")
            elif component == 'unet':
                from diffusers import UNet2DConditionModel
                unet = UNet2DConditionModel.from_pretrained(f"{model_id}", subfolder=component)
                print(f"   ✓ {component}: Loaded successfully")
            elif component == 'vae':
                from diffusers import AutoencoderKL
                vae = AutoencoderKL.from_pretrained(f"{model_id}", subfolder=component)
                print(f"   ✓ {component}: Loaded successfully")
            elif component == 'scheduler':
                # Scheduler usually doesn't have loading issues
                print(f"   - {component}: Skipping (scheduler)")
        except Exception as e:
            print(f"   ✗ {component}: {str(e)[:100]}...")
    
    print()
    
    # 5. Check your environment
    print("5. Environment Information:")
    print(f"   PyTorch version: {torch.__version__}")
    try:
        import diffusers
        print(f"   Diffusers version: {diffusers.__version__}")
    except:
        print("   Diffusers: Not installed or error importing")
    
    try:
        import transformers
        print(f"   Transformers version: {transformers.__version__}")
    except:
        print("   Transformers: Not installed or error importing")
    
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA device: {torch.cuda.get_device_name()}")

if __name__ == "__main__":
    diagnose_pixelgen()


