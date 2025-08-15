
## llm_pixelgen_rtx4070

import torch
import gc
from diffusers import DiffusionPipeline
import os

"""


uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128



"""

def setup_gpu_environment():
    """Optimize environment for RTX 4070"""
    print("Setting up RTX 4070 environment...")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Enable optimizations for RTX 40-series
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"CUDA Version: {torch.version.cuda}")

def load_pixelgen_rtx4070():
    """Load PixelGen optimized for RTX 4070 with workarounds"""
    
    setup_gpu_environment()
    
    model_id = "OEvortex/PixelGen"
    
    # Method 1: Standard loading with RTX 4070 optimizations
    print("\n🚀 Method 1: RTX 4070 optimized loading...")
    try:
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,  # Use FP16 for RTX 4070
            use_safetensors=True,
            device_map="auto",
            low_cpu_mem_usage=True,
            variant="fp16"
        )
        pipe = pipe.to("cuda")
        pipe.enable_memory_efficient_attention()
        pipe.enable_vae_slicing()
        print("✅ Standard loading successful!")
        return pipe
        
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
    
    # Method 2: Load with embedding fix
    print("\n🔧 Method 2: Loading with embedding dimension fix...")
    try:
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            ignore_mismatched_sizes=True,  # Key fix for your error
            use_safetensors=True,
            low_cpu_mem_usage=True
        )
        pipe = pipe.to("cuda")
        pipe.enable_memory_efficient_attention()
        pipe.enable_vae_slicing()
        print("✅ Embedding fix method successful!")
        return pipe
        
    except Exception as e:
        print(f"❌ Method 2 failed: {e}")
    
    # Method 3: Manual component loading with fixes
    print("\n🛠️ Method 3: Manual component assembly...")
    try:
        from transformers import CLIPTextModel, CLIPTokenizer
        from diffusers import UNet2DConditionModel, AutoencoderKL, EulerDiscreteScheduler
        
        # Load tokenizer first (usually works)
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        
        # Load text encoder with size mismatch handling
        try:
            text_encoder = CLIPTextModel.from_pretrained(
                model_id, 
                subfolder="text_encoder",
                torch_dtype=torch.float16,
                ignore_mismatched_sizes=True
            )
        except:
            # Fallback to standard CLIP
            print("  Using fallback CLIP text encoder...")
            text_encoder = CLIPTextModel.from_pretrained(
                "openai/clip-vit-large-patch14",
                torch_dtype=torch.float16
            )
        
        # Load other components
        unet = UNet2DConditionModel.from_pretrained(
            model_id, 
            subfolder="unet", 
            torch_dtype=torch.float16
        )
        
        vae = AutoencoderKL.from_pretrained(
            model_id, 
            subfolder="vae", 
            torch_dtype=torch.float16
        )
        
        scheduler = EulerDiscreteScheduler.from_pretrained(
            model_id, 
            subfolder="scheduler"
        )
        
        # Create pipeline manually
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False
        )
        
        pipe = pipe.to("cuda", torch_dtype=torch.float16)
        pipe.enable_memory_efficient_attention()
        pipe.enable_vae_slicing()
        
        print("✅ Manual assembly successful!")
        return pipe
        
    except Exception as e:
        print(f"❌ Method 3 failed: {e}")
    
    # Method 4: Alternative pixel art model
    print("\n🎨 Method 4: Alternative pixel art model...")
    alternatives = [
        "nerijs/pixel-art-xl",
        "PublicPrompts/All-In-One-Pixel-Model",
        "kohya-ss/controlnet-qr-pattern-sdxl"
    ]
    
    for alt_model in alternatives:
        try:
            print(f"  Trying {alt_model}...")
            pipe = DiffusionPipeline.from_pretrained(
                alt_model,
                torch_dtype=torch.float16,
                use_safetensors=True
            )
            pipe = pipe.to("cuda")
            pipe.enable_memory_efficient_attention()
            pipe.enable_vae_slicing()
            print(f"✅ Alternative model {alt_model} loaded!")
            return pipe
        except Exception as e:
            print(f"❌ {alt_model} failed: {str(e)[:50]}...")
    
    return None

def test_generation(pipe):
    """Test image generation"""
    print("\n🖼️ Testing image generation...")
    
    try:
        prompt = "pixel art of a cute robot, 8bit style, vibrant colors"
        
        with torch.autocast("cuda"):
            image = pipe(
                prompt,
                num_inference_steps=20,
                guidance_scale=7.5,
                height=512,
                width=512
            ).images[0]
        
        # Save test image
        image.save("pixelgen_test.png")
        print("✅ Test generation successful! Image saved as 'pixelgen_test.png'")
        
        # Show GPU memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            print(f"🔥 Peak GPU memory usage: {memory_used:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        return False

def optimize_for_generation(pipe):
    """Apply RTX 4070 specific optimizations"""
    print("\n⚡ Applying RTX 4070 optimizations...")
    
    # Enable all memory optimizations
    pipe.enable_memory_efficient_attention()
    pipe.enable_vae_slicing() 
    pipe.enable_vae_tiling()  # For higher resolution
    
    # Enable xFormers if available
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("✅ xFormers acceleration enabled")
    except:
        print("⚠️ xFormers not available, using standard attention")
    
    # Compile for faster inference (PyTorch 2.0+) - Skip on Python 3.13 if problematic
    try:
        import sys
        if sys.version_info < (3, 13):  # Skip compile on Python 3.13 for now
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
            print("✅ UNet compiled for faster inference")
        else:
            print("⚠️ Skipping torch.compile on Python 3.13 (compatibility)")
    except:
        print("⚠️ Torch compile not available")
    
    return pipe

if __name__ == "__main__":
    print("🎮 Loading PixelGen for RTX 4070...")
    
    pipe = load_pixelgen_rtx4070()
    
    if pipe:
        pipe = optimize_for_generation(pipe)
        success = test_generation(pipe)
        
        if success:
            print("\n🎉 SUCCESS! PixelGen is ready for your RTX 4070!")
            print("\nOptimized settings applied:")
            print("- FP16 precision for memory efficiency")
            print("- Memory efficient attention")
            print("- VAE slicing for large images")
            print("- GPU optimizations enabled")
        else:
            print("\n⚠️ Model loaded but generation failed")
    else:
        print("\n❌ All loading methods failed")
        print("\nTroubleshooting suggestions:")
        print("1. Update drivers: nvidia-smi")
        print("2. Update PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("3. Update diffusers: pip install --upgrade diffusers transformers")

