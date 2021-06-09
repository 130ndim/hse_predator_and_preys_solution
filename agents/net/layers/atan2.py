import math

import torch


class Atan2(torch.nn.Module):
    def __init__(self, dim=-1, max=math.pi):
        super().__init__()
        self.dim = dim
        self.max = max

    def forward(self, x):
        dims = list(range(x.ndim))
        dims.pop(self.dim)
        print(dims, self.dim)
        out = torch.atan2(*x.permute(self.dim, *dims)).unsqueeze(self.dim)
        out /= self.max
        return out