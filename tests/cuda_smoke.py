import torch

A = torch.tensor([
    [1., 2., 3.],
    [4., 5., 6.]
], dtype=torch.float32)

B = torch.tensor([
    [7., 8.],
    [9., 10.],
    [11., 12.]
], dtype=torch.float32)

mult_result_cpu = A @ B

if not torch.accelerator.is_available():
    raise RuntimeError("GPU not available") 
    
A_GPU = A.to(torch.accelerator.current_accelerator())
B_GPU = B.to(torch.accelerator.current_accelerator())
mult_result_gpu = A_GPU @ B_GPU

mult_result_cpu_from_gpu = mult_result_gpu.cpu()

torch.testing.assert_close(mult_result_cpu, mult_result_cpu_from_gpu)

print("Properties of CPU tensor")
print(mult_result_cpu.device)
print(mult_result_cpu.shape)
print(mult_result_cpu.dtype)

print("-----")

print("Properties of GPU tensor")
print(mult_result_gpu.device)
print(mult_result_gpu.shape)
print(mult_result_gpu.dtype)
