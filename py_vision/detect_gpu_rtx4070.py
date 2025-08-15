
### detect_gpu_rtx4070

"""
GPU Detection Test Script for RTX 4070
"""

def test_gpu_detection():
    print("=" * 50)
    print("RTX 4070 Detection Test")
    print("=" * 50)
    
    # Test 1: Basic torch import
    try:
        import torch
        print("✅ PyTorch imported successfully")
        print(f"   PyTorch version: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    # Test 2: CUDA availability
    print(f"\n🔍 CUDA availability: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        print("\n🔧 Troubleshooting steps:")
        print("1. Check if you have CUDA-enabled PyTorch:")
        print("   pip show torch | grep Version")
        print("2. Reinstall with CUDA:")
        print("   uv pip install torch --index-url https://download.pytorch.org/whl/cu121")
        return False
    
    # Test 3: CUDA details
    print(f"   CUDA compiled version: {torch.version.cuda}")
    print(f"   CUDA runtime version: {torch.version.cuda}")
    
    # Test 4: GPU device info
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"\n🎮 GPU devices found: {device_count}")
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"   Device {i}: {props.name}")
            print(f"   Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"   Compute capability: {props.major}.{props.minor}")
            
            # Check if it's RTX 4070
            if "4070" in props.name:
                print("   ✅ RTX 4070 detected!")
            else:
                print(f"   ⚠️  Expected RTX 4070, found: {props.name}")
    
    # Test 5: CUDA operations
    print(f"\n🧪 Testing CUDA operations...")
    try:
        # Create tensors on GPU
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        
        # Perform operation
        z = torch.mm(x, y)
        
        # Check result
        result_cpu = z.cpu()
        print("✅ CUDA matrix multiplication successful")
        print(f"   Result tensor shape: {result_cpu.shape}")
        
        # Memory test
        memory_allocated = torch.cuda.memory_allocated() / 1024**2
        memory_cached = torch.cuda.memory_reserved() / 1024**2
        print(f"   GPU memory allocated: {memory_allocated:.1f} MB")
        print(f"   GPU memory cached: {memory_cached:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ CUDA operation failed: {e}")
        return False

def check_nvidia_driver():
    """Check NVIDIA driver installation"""
    import subprocess
    import sys
    
    print(f"\n🔧 Checking NVIDIA driver...")
    try:
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, 
                              text=True, 
                              timeout=10)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Driver Version' in line:
                    print(f"✅ NVIDIA driver found: {line.strip()}")
                    return True
        else:
            print("❌ nvidia-smi command failed")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found - NVIDIA driver not installed?")
        return False
    except subprocess.TimeoutExpired:
        print("❌ nvidia-smi timed out")
        return False
    except Exception as e:
        print(f"❌ Error checking driver: {e}")
        return False

def main():
    # Check driver first
    driver_ok = check_nvidia_driver()
    
    # Check GPU detection
    gpu_ok = test_gpu_detection()
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    if driver_ok and gpu_ok:
        print("🎉 SUCCESS! RTX 4070 is ready for PyTorch!")
        print("\nYou can now run PixelGen with GPU acceleration.")
    elif not driver_ok:
        print("❌ NVIDIA driver issue detected")
        print("\n🔧 Solutions:")
        print("1. Update NVIDIA drivers from nvidia.com")
        print("2. Restart your computer after driver installation")
        print("3. Check Device Manager for GPU status")
    elif not gpu_ok:
        print("❌ PyTorch CUDA issue detected")
        print("\n🔧 Solutions:")
        print("1. Reinstall PyTorch with CUDA:")
        print("   uv pip uninstall torch torchvision torchaudio")
        print("   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("2. Try CUDA 11.8 if 12.1 doesn't work:")
        print("   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

if __name__ == "__main__":
    main()


