import torch
import time

# Create a large matrix operation that's slow on CPU but fast on GPU
def test_computation(device_name):
    device = torch.device(device_name)
    
    # Create large matrices
    size = 5000
    matrix1 = torch.randn(size, size, device=device)
    matrix2 = torch.randn(size, size, device=device)
    
    # Time the matrix multiplication
    start = time.time()
    result = torch.matmul(matrix1, matrix2)
    # Force completion of GPU operations
    if device_name == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"Device: {device_name}")
    print(f"Matrix shape: {size}x{size}")
    print(f"Time taken: {elapsed:.4f} seconds")
    print(f"Result shape: {result.shape}")
    print(f"Result is on {result.device}")
    
    return elapsed

# Run on CPU
cpu_time = test_computation('cpu')

# Run on GPU if available
if torch.cuda.is_available():
    gpu_time = test_computation('cuda')
    print(f"\nGPU is {cpu_time/gpu_time:.1f}x faster than CPU")
else:
    print("\nCUDA not available")