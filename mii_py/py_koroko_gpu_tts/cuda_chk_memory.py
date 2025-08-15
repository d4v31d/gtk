
### cuda_chk_memory

import torch
import sys
import gc
import time

def check_cuda_memory():
    """Check available CUDA memory and return stats"""
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return None
    
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    free_memory = total_memory - allocated_memory
    
    print(f"Total GPU memory: {total_memory / 1024**3:.2f} GB")
    print(f"Allocated memory: {allocated_memory / 1024**3:.2f} GB")
    print(f"Free memory: {free_memory / 1024**3:.2f} GB")
    
    return {
        'total': total_memory,
        'allocated': allocated_memory,
        'free': free_memory
    }

def require_cuda_memory(required_gb):
    """Exit if insufficient CUDA memory is available"""
    memory_stats = check_cuda_memory()
    if memory_stats is None:
        print("CUDA not available. Exiting...")
        sys.exit(1)
    
    required_bytes = required_gb * 1024**3
    if memory_stats['free'] < required_bytes:
        print(f"Insufficient GPU memory. Required: {required_gb} GB, Available: {memory_stats['free'] / 1024**3:.2f} GB")
        print("Exiting due to insufficient CUDA memory...")
        sys.exit(1)
    else:
        print(f"Sufficient GPU memory available. Continuing...")

def safe_cuda_operation(operation_func, *args, **kwargs):
    """Safely execute CUDA operations with memory error handling"""
    try:
        return operation_func(*args, **kwargs)
    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA Out of Memory Error: {e}")
        print("Attempting to free GPU memory...")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        gc.collect()
        
        # Check memory after cleanup
        check_cuda_memory()
        
        print("Exiting due to CUDA out of memory error...")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

def monitor_memory_usage(func):
    """Decorator to monitor memory usage of a function"""
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            print("Memory before operation:")
            check_cuda_memory()
            
            try:
                result = func(*args, **kwargs)
                print("Memory after operation:")
                check_cuda_memory()
                return result
            except torch.cuda.OutOfMemoryError:
                print("Out of memory during operation!")
                torch.cuda.empty_cache()
                gc.collect()
                sys.exit(1)
        else:
            return func(*args, **kwargs)
    return wrapper

# Example usage:

if __name__ == "__main__":

    print()
    print()
    print("-" * 70 )
    ts_begin = time.strftime("%Y%m%d%H%M%S") 
    print(f"ts_begin: {ts_begin}")
    print()

    print("Checking CUDA memory...")
    print()


    # Method 1: Check before starting
    require_cuda_memory(2.0)  # Require at least 2GB free
    
    # Method 2: Safe operation wrapper
    def create_large_tensor():
        return torch.randn(10000, 10000).cuda()
    
    tensor = safe_cuda_operation(create_large_tensor)
    
    # Method 3: Using decorator
    @monitor_memory_usage
    def train_model():
        # Your training code here
        model = torch.nn.Linear(1000, 1000).cuda()
        return model
    
    model = train_model()
    
    print("All operations completed successfully!")


    ts_end = time.strftime("%Y%m%d%H%M%S")
    print()
    print(f"ts_end: {ts_end}")
    print()
    print()
    print()


