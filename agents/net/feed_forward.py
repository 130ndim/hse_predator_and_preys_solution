from typing import Sequence

from torch import Tensor, nn


class FFN(nn.Module):
    def __init__(self, hidden_sizes: Sequence[int], act=nn.LeakyReLU(0.2, inplace=False)):
        super().__init__()
        seq = []
        for in_, out_ in zip(hidden_sizes[:-2], hidden_sizes[1:-1]):
            seq += [nn.Linear(in_, out_), act]
        seq += [nn.Linear(hidden_sizes[-2], hidden_sizes[-1])]

        self.seq = nn.Sequential(*seq)

    def forward(self, state: Tensor):
        out = self.seq(state)
        return out
