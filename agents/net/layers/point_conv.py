from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import Sequential as Seq, Linear as Lin, Conv2d, ReLU

AGGRS = {
    'min': lambda x: torch.min(x, dim=-2)[0],
    'max': lambda x: torch.max(x, dim=-2)[0],
    'mean': lambda x: torch.mean(x, dim=-2),
    'std': lambda x: torch.std(x, dim=-2)
}


class PointConv(nn.Module):
    def __init__(self, hidden_size, pos_size: int = 2, aggrs=('min', 'max', 'mean')):
        super().__init__()
        self.lin = Lin(hidden_size, hidden_size)
        self.act = nn.ELU()

        self.aggrs = [AGGRS[aggr] for aggr in aggrs]

        self.nn1 = Seq(Lin(hidden_size + pos_size, hidden_size * 2), ReLU(), Lin(hidden_size * 2, hidden_size))
        self.nn2 = Seq(Lin(hidden_size * len(aggrs), hidden_size * 2), ReLU(), Lin(hidden_size * 2, hidden_size))

    def forward(
            self,
            x_pred: Tensor, x_prey: Tensor, x_obst: Tensor,
            pos_pred: Tensor, pos_prey: Tensor, pos_obst: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        N = (x_pred.size(1), x_prey.size(1), x_obst.size(1))  # type: ignore

        x = torch.cat([x_pred, x_prey, x_obst], dim=1)
        pos = torch.cat([pos_pred, pos_prey, pos_obst], dim=1)

        diff = (pos.unsqueeze(2) - pos.unsqueeze(1))
        m = torch.cat([x.unsqueeze(1).repeat(1, sum(N), 1, 1), diff], dim=-1)
        m = self.nn1(m)
        m = torch.cat([aggr(m) for aggr in self.aggrs], dim=-1)

        x = x + self.nn2(m)

        out_pred, out_prey, out_obst = x.split(N, dim=1)
        return out_pred, out_prey, out_obst
