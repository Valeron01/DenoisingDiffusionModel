import torch


a = torch.randn(10000)
indices = torch.randint(0, a.shape[0], (100,))

print(a.gather(-1, indices) == a[indices])

