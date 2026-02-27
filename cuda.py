import cupy as cp
import numpy as np
import time

# Matrix multiplication benchmark: CPU vs GPU

size = 4096

# CPU computation
a_cpu = np.random.rand(size, size).astype(np.float32)
b_cpu = np.random.rand(size, size).astype(np.float32)

start = time.time()
c_cpu = np.matmul(a_cpu, b_cpu)
cpu_time = time.time() - start
print(f"CPU time: {cpu_time:.3f}s")

# GPU computation
a_gpu = cp.asarray(a_cpu)  # Transfer to GPU
b_gpu = cp.asarray(b_cpu)

cp.cuda.Stream.null.synchronize()  # Ensure GPU is ready
start = time.time()
c_gpu = cp.matmul(a_gpu, b_gpu)
cp.cuda.Stream.null.synchronize()  # Wait for GPU to finish
gpu_time = time.time() - start
print(f"GPU time: {gpu_time:.3f}s")

print(f"Speedup: {cpu_time / gpu_time:.1f}x faster on GPU")

# Verify results match
c_gpu_back = cp.asnumpy(c_gpu)  # Transfer result back to CPU
print(f"Results match: {np.allclose(c_cpu, c_gpu_back, atol=1e-3)}")
