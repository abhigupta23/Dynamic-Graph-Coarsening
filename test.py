import torch

# Create a matrix on the GPU
A = torch.randn(4, 3, device='cuda')

# Compute the pseudo-inverse using torch.linalg.pinv
A_pinv = torch.linalg.pinv(A)

print(A_pinv)