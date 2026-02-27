import torch
import time

size = 4096

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

a = torch.rand(size, size, dtype=torch.float32)
b = torch.rand(size, size, dtype=torch.float32)

# CPU
start = time.time()
c_cpu = torch.matmul(a, b)
print(f"CPU time: {time.time() - start:.3f}s")

# GPU
a_gpu = a.to(device)
b_gpu = b.to(device)

torch.cuda.synchronize()
start = time.time()
c_gpu = torch.matmul(a_gpu, b_gpu)
torch.cuda.synchronize()
print(f"GPU time: {time.time() - start:.3f}s")
