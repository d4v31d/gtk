#!/usr/bin/env python3
"""
LLM GPU Testing Suite
Comprehensive testing framework for evaluating LLM performance on GPU
"""

import torch
import time
import psutil
import gc
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

class GPUTester:
    def __init__(self):
        self.device = self._setup_device()
        self.results = []
        
    def _setup_device(self) -> torch.device:
        """Setup and verify GPU device"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return device
        else:
            print("⚠️  CUDA not available, using CPU")
            return torch.device("cpu")
    
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "total_gb": total,
                "free_gb": total - reserved
            }
        return {"allocated_gb": 0, "reserved_gb": 0, "total_gb": 0, "free_gb": 0}
    
    def test_tensor_operations(self, size: int = 10000) -> Dict[str, float]:
        """Test basic tensor operations on GPU"""
        print(f"\n🧪 Testing tensor operations (size: {size}x{size})")
        
        # Matrix multiplication test
        start_time = time.time()
        a = torch.randn(size, size, device=self.device)
        b = torch.randn(size, size, device=self.device)
        c = torch.matmul(a, b)
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        matmul_time = time.time() - start_time
        
        # Element-wise operations
        start_time = time.time()
        d = torch.relu(c)
        e = torch.softmax(d, dim=1)
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        elementwise_time = time.time() - start_time
        
        # Memory cleanup
        del a, b, c, d, e
        torch.cuda.empty_cache() if self.device.type == 'cuda' else None
        
        results = {
            "matmul_time": matmul_time,
            "elementwise_time": elementwise_time,
            "total_time": matmul_time + elementwise_time
        }
        
        print(f"  Matrix multiplication: {matmul_time:.3f}s")
        print(f"  Element-wise ops: {elementwise_time:.3f}s")
        
        return results
    
    def test_transformer_layer(self, batch_size: int = 32, seq_len: int = 512, d_model: int = 768) -> Dict[str, float]:
        """Test transformer-like operations"""
        print(f"\n🧪 Testing transformer layer (batch={batch_size}, seq_len={seq_len}, d_model={d_model})")
        
        # Simulate attention computation
        start_time = time.time()
        
        # Input embeddings
        x = torch.randn(batch_size, seq_len, d_model, device=self.device)
        
        # Multi-head attention simulation
        num_heads = 12
        head_dim = d_model // num_heads
        
        # Q, K, V projections
        q = torch.randn(batch_size, seq_len, d_model, device=self.device)
        k = torch.randn(batch_size, seq_len, d_model, device=self.device)
        v = torch.randn(batch_size, seq_len, d_model, device=self.device)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Feed forward simulation
        ff_hidden = torch.relu(torch.randn(batch_size, seq_len, d_model * 4, device=self.device))
        ff_output = torch.randn(batch_size, seq_len, d_model, device=self.device)
        
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        total_time = time.time() - start_time
        
        # Memory cleanup
        del x, q, k, v, scores, attn_weights, attn_output, ff_hidden, ff_output
        torch.cuda.empty_cache() if self.device.type == 'cuda' else None
        
        print(f"  Transformer layer time: {total_time:.3f}s")
        
        return {"transformer_time": total_time}
    
    def test_memory_scaling(self, sizes: List[int] = None) -> Dict[str, List[float]]:
        """Test memory usage scaling with different tensor sizes"""
        if sizes is None:
            sizes = [1000, 2000, 5000, 8000, 10000]
        
        print(f"\n🧪 Testing memory scaling")
        
        memory_usage = []
        computation_times = []
        
        for size in sizes:
            try:
                # Clear memory
                torch.cuda.empty_cache() if self.device.type == 'cuda' else None
                gc.collect()
                
                start_mem = self.get_gpu_memory_info()["allocated_gb"]
                
                # Create tensors
                start_time = time.time()
                a = torch.randn(size, size, device=self.device)
                b = torch.randn(size, size, device=self.device)
                c = torch.matmul(a, b)
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                comp_time = time.time() - start_time
                
                end_mem = self.get_gpu_memory_info()["allocated_gb"]
                mem_used = end_mem - start_mem
                
                memory_usage.append(mem_used)
                computation_times.append(comp_time)
                
                print(f"  Size {size}: {mem_used:.2f} GB, {comp_time:.3f}s")
                
                # Cleanup
                del a, b, c
                torch.cuda.empty_cache() if self.device.type == 'cuda' else None
                
            except RuntimeError as e:
                print(f"  Size {size}: Out of memory - {str(e)}")
                break
        
        return {
            "sizes": sizes[:len(memory_usage)],
            "memory_usage": memory_usage,
            "computation_times": computation_times
        }
    
    def benchmark_throughput(self, duration: int = 10) -> Dict[str, float]:
        """Benchmark throughput over a specified duration"""
        print(f"\n🧪 Benchmarking throughput for {duration} seconds")
        
        operations = 0
        start_time = time.time()
        
        size = 2000  # Moderate size for sustained testing
        
        while time.time() - start_time < duration:
            try:
                a = torch.randn(size, size, device=self.device)
                b = torch.randn(size, size, device=self.device)
                c = torch.matmul(a, b)
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                
                operations += 1
                
                # Cleanup every 10 operations
                if operations % 10 == 0:
                    del a, b, c
                    torch.cuda.empty_cache() if self.device.type == 'cuda' else None
                
            except RuntimeError:
                print("  Out of memory during throughput test")
                break
        
        total_time = time.time() - start_time
        ops_per_second = operations / total_time
        
        print(f"  Operations completed: {operations}")
        print(f"  Operations per second: {ops_per_second:.2f}")
        
        return {
            "total_operations": operations,
            "duration": total_time,
            "ops_per_second": ops_per_second
        }
    
    def stress_test(self, duration: int = 30) -> Dict[str, any]:
        """Run a stress test to check stability"""
        print(f"\n🧪 Running stress test for {duration} seconds")
        
        start_time = time.time()
        operations = 0
        errors = 0
        max_memory = 0
        
        while time.time() - start_time < duration:
            try:
                # Variable size operations
                size = np.random.randint(1000, 5000)
                a = torch.randn(size, size, device=self.device)
                b = torch.randn(size, size, device=self.device)
                
                # Random operations
                if np.random.random() > 0.5:
                    c = torch.matmul(a, b)
                else:
                    c = a + b
                
                # Track memory
                current_mem = self.get_gpu_memory_info()["allocated_gb"]
                max_memory = max(max_memory, current_mem)
                
                operations += 1
                
                # Periodic cleanup
                if operations % 20 == 0:
                    del a, b, c
                    torch.cuda.empty_cache() if self.device.type == 'cuda' else None
                
            except RuntimeError as e:
                errors += 1
                print(f"  Error {errors}: {str(e)}")
                torch.cuda.empty_cache() if self.device.type == 'cuda' else None
        
        total_time = time.time() - start_time
        
        results = {
            "duration": total_time,
            "operations": operations,
            "errors": errors,
            "max_memory_gb": max_memory,
            "success_rate": (operations / (operations + errors)) * 100 if operations + errors > 0 else 0
        }
        
        print(f"  Operations: {operations}, Errors: {errors}")
        print(f"  Success rate: {results['success_rate']:.1f}%")
        print(f"  Max memory used: {max_memory:.2f} GB")
        
        return results
    
    def run_full_test_suite(self) -> Dict[str, any]:
        """Run complete test suite"""
        print("=" * 50)
        print("🚀 Starting LLM GPU Test Suite")
        print("=" * 50)
        
        # System info
        print(f"Device: {self.device}")
        print(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name()}")
            mem_info = self.get_gpu_memory_info()
            print(f"GPU memory: {mem_info['total_gb']:.1f} GB total, {mem_info['free_gb']:.1f} GB free")
        
        # Run tests
        results = {
            "timestamp": datetime.now().isoformat(),
            "device": str(self.device),
            "pytorch_version": torch.__version__,
        }
        
        try:
            results["tensor_ops"] = self.test_tensor_operations()
            results["transformer"] = self.test_transformer_layer()
            results["memory_scaling"] = self.test_memory_scaling()
            results["throughput"] = self.benchmark_throughput()
            results["stress_test"] = self.stress_test()
            
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")
            results["error"] = str(e)
        
        finally:
            # Final cleanup
            torch.cuda.empty_cache() if self.device.type == 'cuda' else None
            gc.collect()
        
        print("\n" + "=" * 50)
        print("✅ Test suite completed!")
        print("=" * 50)
        
        return results

def main():
    """Main function to run the GPU test suite"""
    tester = GPUTester()
    results = tester.run_full_test_suite()
    
    # Optional: Save results to file
    # import json
    # with open(f"gpu_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
    #     json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    main()



"""

--------------------------------------------------------------------------------------------------------
250805 1658

⚠️  CUDA not available, using CPU
==================================================
🚀 Starting LLM GPU Test Suite
==================================================
Device: cpu
PyTorch version: 2.7.1+cpu

🧪 Testing tensor operations (size: 10000x10000)
  Matrix multiplication: 2.869s
  Element-wise ops: 0.180s

🧪 Testing transformer layer (batch=32, seq_len=512, d_model=768)
  Transformer layer time: 0.723s

🧪 Testing memory scaling
  Size 1000: 0.00 GB, 0.012s
  Size 2000: 0.00 GB, 0.056s
  Size 5000: 0.00 GB, 0.448s
  Size 8000: 0.00 GB, 1.499s
  Size 10000: 0.00 GB, 2.700s

🧪 Benchmarking throughput for 10 seconds
  Operations completed: 161
  Operations per second: 16.04

🧪 Running stress test for 30 seconds
  Operations: 170, Errors: 0
  Success rate: 100.0%
  Max memory used: 0.00 GB

==================================================
✅ Test suite completed!
==================================================


"""

