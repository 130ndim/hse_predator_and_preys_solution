from typing import Sequence

from torch import Tensor, nn
from torch.nn import init


class FFN(nn.Module):
    def __init__(self, hidden_sizes: Sequence[int], act=nn.ReLU()):
        super().__init__()
        seq = []
        for in_, out_ in zip(hidden_sizes[:-2], hidden_sizes[1:-1]):
            seq += [nn.LayerNorm(in_, elementwise_affine=False), nn.Linear(in_, out_), act]
        seq += [nn.LayerNorm(hidden_sizes[-2], elementwise_affine=False),
                nn.Linear(hidden_sizes[-2], hidden_sizes[-1])]

        self.seq = nn.Sequential(*seq)

        self.reset_parameters()

    def reset_parameters(self):
        for n, p in self.seq.named_parameters():
            if 'weight' in n:
                init.xavier_uniform_(p)
            elif 'bias' in n:
                init.zeros_(p.data)

    def forward(self, state: Tensor):
        out = self.seq(state)
        return out
