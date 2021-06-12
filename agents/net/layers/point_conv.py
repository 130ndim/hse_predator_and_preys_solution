from typing import Optional, Tuple

import torch
from torch import BoolTensor, Tensor, nn
from torch.nn import Sequential as Seq, Linear as Lin, Conv2d, LeakyReLU as LReLU, init

AGGRS = {
    'min': lambda x, mask: x.masked_fill(mask, float('inf')).min(dim=-2)[0],
    'max': lambda x, mask: x.masked_fill(mask, -float('inf')).max(dim=-2)[0],
    'mean': lambda x, mask: (x * ~mask).sum(dim=-2) / (~mask).sum(dim=-2),
    'std': lambda x: torch.std(x, dim=-2)
}


class PointConv(nn.Module):
    def __init__(self, hidden_size, pos_size: int = 2, aggrs=('min', 'max', 'mean')):
        super().__init__()
        self.aggrs = [AGGRS[aggr] for aggr in aggrs]

        self.nn1 = Seq(Lin(hidden_size + pos_size, hidden_size * 2), LReLU(0.15), Lin(hidden_size * 2, hidden_size))
        self.nn2 = Seq(Lin(hidden_size * len(aggrs), hidden_size * 2), LReLU(0.15), Lin(hidden_size * 2, hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        for n, p in list(self.nn1.named_parameters()) + list(self.nn2.named_parameters()):
            if 'weight' in n:
                init.xavier_uniform_(p)
            elif 'bias' in n:
                init.zeros_(p)

    def forward(
            self,
            x_pred: Tensor, x_prey: Tensor, x_obst: Tensor,
            pos_pred: Tensor, pos_prey: Tensor, pos_obst: Tensor,
            prey_is_alive: Optional[BoolTensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        N = (x_pred.size(1), x_prey.size(1), x_obst.size(1))  # type: ignore

        x = torch.cat([x_pred, x_prey, x_obst], dim=1)
        pos = torch.cat([pos_pred, pos_prey, pos_obst], dim=1)

        diff = (pos.unsqueeze(2) - pos.unsqueeze(1))
        if prey_is_alive is not None:
            mask = ~torch.cat([pos_pred.new_ones(pos_pred.size()[:2], dtype=torch.bool),
                               prey_is_alive.bool(),
                               pos_obst.new_ones(pos_obst.size()[:2], dtype=torch.bool)], dim=1)
            mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        else:
            mask = pos_pred.new_zeros(diff.size()[:3], dtype=torch.bool)
        mask = mask.unsqueeze(-1)

        m = torch.cat([x.unsqueeze(1).repeat(1, sum(N), 1, 1), diff], dim=-1)
        m = self.nn1(m)
        m = torch.cat([aggr(m, mask) for aggr in self.aggrs], dim=-1)

        x = x + self.nn2(m)

        out_pred, out_prey, out_obst = x.split(N, dim=1)
        return out_pred, out_prey, out_obst


class RPointConv(nn.Module):
    def __init__(self, hidden_size, pos_size: int = 2, aggrs=('min', 'max', 'mean')):
        super().__init__()
        self.aggrs = [AGGRS[aggr] for aggr in aggrs]

        self.nn1 = Seq(Lin(3 * len(aggrs) * (hidden_size + pos_size), hidden_size * 2), LReLU(0.15), Lin(hidden_size * 2, hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        for n, p in list(self.nn1.named_parameters()) + list(self.nn2.named_parameters()):
            if 'weight' in n:
                init.xavier_uniform_(p)
            elif 'bias' in n:
                init.zeros_(p)

    def forward(
            self,
            x: Tensor,
            x_pos: Tensor,
            x_pred: Tensor, x_prey: Tensor, x_obst: Tensor,
            pos_pred: Tensor, pos_prey: Tensor, pos_obst: Tensor,
            prey_is_alive: Optional[BoolTensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        N = (x_pred.size(1), x_prey.size(1), x_obst.size(1))  # type: ignore

        x = torch.cat([x_pred, x_prey, x_obst], dim=1)
        pos = torch.cat([pos_pred, pos_prey, pos_obst], dim=1)

        diff = (pos.unsqueeze(2) - pos.unsqueeze(1))
        if prey_is_alive is not None:
            mask = ~torch.cat([pos_pred.new_ones(pos_pred.size()[:2], dtype=torch.bool),
                               prey_is_alive.bool(),
                               pos_obst.new_ones(pos_obst.size()[:2], dtype=torch.bool)], dim=1)
            mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        else:
            mask = pos_pred.new_zeros(diff.size()[:3], dtype=torch.bool)
        mask = mask.unsqueeze(-1)

        m = torch.cat([x.unsqueeze(1).repeat(1, sum(N), 1, 1), diff], dim=-1)
        m = self.nn1(m)
        m = torch.cat([aggr(m, mask) for aggr in self.aggrs], dim=-1)

        x = x + self.nn2(m)

        out_pred, out_prey, out_obst = x.split(N, dim=1)
        return out_pred, out_prey, out_obst

