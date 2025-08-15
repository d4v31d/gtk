
### llm_pixelgen_02
### claude.ai version

# Alternative methods to load PixelGen
import torch
from diffusers import DiffusionPipeline
import os

def method1_manual_component_loading():
    """Load components individually and build pipeline manually"""
    print("Method 1: Manual component loading...")
    try:
        from transformers import CLIPTextModel, CLIPTokenizer
        from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
        
        model_id = "OEvortex/PixelGen"
        
        # Load each component separately
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        
        # Try different text encoder approaches
        try:
            text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        except RuntimeError as e:
            print(f"Standard text_encoder failed: {e}")
            # Try with ignore_mismatched_sizes
            text_encoder = CLIPTextModel.from_pretrained(
                model_id, 
                subfolder="text_encoder",
                ignore_mismatched_sizes=True
            )
        
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        
        print("✓ All components loaded successfully!")
        return tokenizer, text_encoder, unet, vae, scheduler
        
    except Exception as e:
        print(f"✗ Method 1 failed: {e}")
        return None

def method2_force_local_files():
    """Force using local files only"""
    print("Method 2: Force local files...")
    try:
        pipe = DiffusionPipeline.from_pretrained(
            "OEvortex/PixelGen",
            local_files_only=True,
            force_download=False
        )
        return pipe
    except Exception as e:
        print(f"✗ Method 2 failed: {e}")
        return None

def method3_different_dtype():
    """Try different data types"""
    print("Method 3: Different data types...")
    dtypes = [torch.float32, torch.float16]
    
    for dtype in dtypes:
        try:
            print(f"  Trying {dtype}...")
            pipe = DiffusionPipeline.from_pretrained(
                "OEvortex/PixelGen",
                torch_dtype=dtype,
                variant="fp16" if dtype == torch.float16 else None
            )
            print(f"✓ Success with {dtype}!")
            return pipe
        except Exception as e:
            print(f"✗ Failed with {dtype}: {str(e)[:50]}...")
    
    return None

def method4_custom_text_encoder():
    """Try with a compatible text encoder"""
    print("Method 4: Custom text encoder replacement...")
    try:
        from transformers import CLIPTextModel, CLIPTokenizer
        from diffusers import StableDiffusionPipeline
        
        # Use a known working text encoder
        backup_text_encoder = "openai/clip-vit-base-patch32"
        
        # Load other components from PixelGen
        model_id = "OEvortex/PixelGen"
        
        tokenizer = CLIPTokenizer.from_pretrained(backup_text_encoder)
        text_encoder = CLIPTextModel.from_pretrained(backup_text_encoder)
        
        # Try to load other components
        from diffusers import UNet2DConditionModel, AutoencoderKL, PNDMScheduler
        
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae") 
        scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")
        
        print("✓ Custom text encoder method successful!")
        return tokenizer, text_encoder, unet, vae, scheduler
        
    except Exception as e:
        print(f"✗ Method 4 failed: {e}")
        return None

def method5_similar_model():
    """Try a similar pixel art model as alternative"""
    print("Method 5: Alternative pixel art models...")
    
    alternatives = [
        "nerijs/pixel-art-xl",
        "kohya-ss/controlnet-qr-pattern-sdxl",
        "stabilityai/stable-diffusion-2-1"
    ]
    
    for alt_model in alternatives:
        try:
            print(f"  Trying {alt_model}...")
            pipe = DiffusionPipeline.from_pretrained(alt_model)
            print(f"✓ Alternative model {alt_model} loaded successfully!")
            return pipe
        except Exception as e:
            print(f"✗ {alt_model} failed: {str(e)[:50]}...")
    
    return None

# Main execution
if __name__ == "__main__":
    print("Trying alternative loading methods for PixelGen...\n")
    
    methods = [
        #method1_manual_component_loading,
        #method2_force_local_files, 
        #method3_different_dtype,
        #method4_custom_text_encoder,
        method5_similar_model
    ]
    
    for i, method in enumerate(methods, 1):
        result = method()
        if result is not None:
            print(f"\n🎉 SUCCESS with Method {i}!")
            break
        print()
    else:
        print("\n❌ All methods failed. Please run the diagnostic script first.")

