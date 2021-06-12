from typing import Sequence

from torch import Tensor, nn
from torch.nn import init

from ..utils import reset


class FFN(nn.Module):
    def __init__(self, hidden_sizes: Sequence[int], act=nn.LeakyReLU(0.1), layer_norm=True):
        super().__init__()
        seq = []
        for in_, out_ in zip(hidden_sizes[:-2], hidden_sizes[1:-1]):
            if layer_norm:
                seq += [nn.LayerNorm(in_, elementwise_affine=True)]
            seq += [nn.Linear(in_, out_), act]
        if layer_norm:
            seq += [nn.LayerNorm(hidden_sizes[-2], elementwise_affine=True)]
        seq += [nn.Linear(hidden_sizes[-2], hidden_sizes[-1])]

        self.seq = nn.Sequential(*seq)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.seq)

    def forward(self, state: Tensor):
        out = self.seq(state)
        return out
