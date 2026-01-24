import torch
import time

use_gpu = torch.cuda.is_available()

if use_gpu:
    print("CUDA is available.")
    device = torch.device("cuda")
else:
    print("CUDA is not available.")
    device = torch.device("cpu")

x = torch.rand(10000, 10000, device=device)
y = torch.rand(10000, 10000, device=device)

if use_gpu:
    start_time_gpu = time.time()
    z = torch.matmul(x, y)
    end_time_gpu = time.time()
    elapsed_time_gpu = end_time_gpu - start_time_gpu
    print(f"Time taken for operation with GPU: {elapsed_time_gpu:.5f} seconds")

start_time_cpu = time.time()
z = torch.matmul(x, y).to("cpu")
end_time_cpu = time.time()
elapsed_time_cpu = end_time_cpu - start_time_cpu
print(f"Time taken for operation with CPU: {elapsed_time_cpu:.5f} seconds")