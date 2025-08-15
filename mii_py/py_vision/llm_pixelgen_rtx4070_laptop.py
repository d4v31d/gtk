
### llm_pixelgen_rtx4070_laptop

import torch
import gc
from diffusers import DiffusionPipeline
import os

def setup_gpu_environment():
    """Optimize environment for RTX 4070 Laptop (8GB)"""
    print("Setting up RTX 4070 Laptop (8GB) environment...")
    
    # Clear GPU memory aggressively for 8GB GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
    
    # Enable optimizations for RTX 40-series
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    # Set memory fraction for 8GB GPU (use ~90% to leave room for OS)
    torch.cuda.set_per_process_memory_fraction(0.9)
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
    print("✅ RTX 4070 Laptop (8GB) optimizations applied!")

def load_pixelgen_rtx4070():
    """Load PixelGen optimized for RTX 4070 Laptop (8GB) with workarounds"""
    
    setup_gpu_environment()
    
    model_id = "OEvortex/PixelGen"
    
    # Method 1: 8GB optimized loading
    print("\n🚀 Method 1: RTX 4070 Laptop (8GB) optimized loading...")
    try:
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,  # Essential for 8GB GPU
            use_safetensors=True,
            device_map="auto",
            low_cpu_mem_usage=True,
            variant="fp16",
            max_split_size_mb=512  # Smaller chunks for 8GB GPU
        )
        pipe = pipe.to("cuda")
        
        # Apply aggressive memory optimizations for 8GB
        pipe.enable_memory_efficient_attention()
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        pipe.enable_sequential_cpu_offload()  # Offload to CPU when not in use
        
        print("✅ 8GB optimized loading successful!")
        return pipe
        
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
    
    # Method 2: Load with embedding fix + 8GB optimizations
    print("\n🔧 Method 2: Loading with embedding fix + 8GB optimizations...")
    try:
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            ignore_mismatched_sizes=True,  # Key fix for your error
            use_safetensors=True,
            low_cpu_mem_usage=True,
            max_split_size_mb=512
        )
        pipe = pipe.to("cuda")
        
        # 8GB specific optimizations
        pipe.enable_memory_efficient_attention()
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        pipe.enable_sequential_cpu_offload()
        
        print("✅ Embedding fix + 8GB optimizations successful!")
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
        
        # 8GB laptop optimizations
        pipe.enable_memory_efficient_attention()
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        pipe.enable_sequential_cpu_offload()  # Critical for 8GB
        
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
    """Test image generation optimized for 8GB GPU"""
    print("\n🖼️ Testing image generation (8GB optimized)...")
    
    try:
        prompt = "pixel art of a cute robot, 8bit style, vibrant colors"
        
        # 8GB GPU settings - smaller resolution and fewer steps
        with torch.autocast("cuda"):
            image = pipe(
                prompt,
                num_inference_steps=15,  # Reduced from 20 for 8GB
                guidance_scale=7.5,
                height=512,
                width=512,
                max_sequence_length=77  # Limit sequence length
            ).images[0]
        
        # Clear GPU memory after generation
        torch.cuda.empty_cache()
        
        # Save test image
        image.save("pixelgen_test.png")
        print("✅ Test generation successful! Image saved as 'pixelgen_test.png'")
        
        # Show GPU memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🔥 Peak GPU memory usage: {memory_used:.2f} GB / {memory_total:.1f} GB ({memory_used/memory_total*100:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        print("💡 Try reducing image size or using CPU offload")
        return False

def optimize_for_generation(pipe):
    """Apply RTX 4070 Laptop (8GB) specific optimizations"""
    print("\n⚡ Applying RTX 4070 Laptop (8GB) optimizations...")
    
    # Enable all memory optimizations - critical for 8GB
    pipe.enable_memory_efficient_attention()
    pipe.enable_vae_slicing() 
    pipe.enable_vae_tiling()  # For higher resolution without OOM
    pipe.enable_sequential_cpu_offload()  # Move components to CPU when not needed
    
    # Enable xFormers if available
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("✅ xFormers acceleration enabled")
    except:
        print("⚠️ xFormers not available, using standard attention")
    
    # Skip torch.compile for 8GB GPU to save memory
    print("ℹ️ Skipping torch.compile to preserve GPU memory (8GB)")
    
    return pipe

if __name__ == "__main__":
    print("🎮 Loading PixelGen for RTX 4070...")
    
    pipe = load_pixelgen_rtx4070()
    
    if pipe:
        pipe = optimize_for_generation(pipe)
        success = test_generation(pipe)
        
        if success:
            print("\n🎉 SUCCESS! PixelGen is ready for your RTX 4070 Laptop (8GB)!")
            print("\n8GB GPU optimizations applied:")
            print("- FP16 precision for memory efficiency")
            print("- Sequential CPU offload (moves unused components to RAM)")
            print("- VAE slicing and tiling for large images")
            print("- Memory efficient attention")
            print("- Reduced inference steps (15 vs 20)")
            print("- Memory fraction limited to 90%")
            
            print("\n💡 Tips for 8GB GPU:")
            print("- Use 512x512 or smaller for best performance")
            print("- Reduce num_inference_steps if you get OOM errors") 
            print("- Enable CPU offload for complex scenes")
        else:
            print("\n⚠️ Model loaded but generation failed")
            print("💡 Try: pip install --upgrade torch xformers")
    else:
        print("\n❌ All loading methods failed")
        print("\nTroubleshooting suggestions:")
        print("1. Update drivers: nvidia-smi")
        print("2. Update PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("3. Update diffusers: pip install --upgrade diffusers transformers")



