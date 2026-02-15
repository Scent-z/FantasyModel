import torch

dim_index = torch.arange(0, 8, 2, device="cuda").float()
print(dim_index)
a = torch.arange(8)
print(a)