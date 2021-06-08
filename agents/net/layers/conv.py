from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import Sequential as Seq, Linear as Lin, Conv2d


class Embedding(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.xy_lin = Seq(Lin(2, hidden_size), nn.LeakyReLU(0.1), Lin(hidden_size, hidden_size))
        self.r_lin = Seq(Lin(1, hidden_size), nn.LeakyReLU(0.1), Lin(hidden_size, hidden_size))

        self.entity_emb = nn.Embedding(3, hidden_size)

        self.xy_bn = nn.BatchNorm1d(2)
        self.r_bn = nn.BatchNorm1d(1)

    def forward(self, pos_pred: Tensor, pos_prey: Tensor, x_obst: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        B, N, _ = pos_pred.size()
        pos_pred = self.xy_bn(pos_pred.view(-1, 2)).view_as(pos_pred)
        pos_prey = self.xy_bn(pos_prey.view(-1, 2)).view_as(pos_prey)

        pos_obst, r_obst = x_obst.split((2, 1), dim=-1)
        pos_obst = self.xy_bn(pos_obst.view(-1, 2)).view_as(pos_obst)
        r_obst = self.r_bn(r_obst.view(-1, 1)).view_as(r_obst)

        out_pred = self.xy_lin(pos_pred) + self.entity_emb.weight[0]
        out_prey = self.xy_lin(pos_prey) + self.entity_emb.weight[1]
        out_obst = self.xy_lin(pos_obst) + self.r_lin(r_obst) + self.entity_emb.weight[2]

        return out_pred, out_prey, out_obst


class Conv(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.lin = Lin(hidden_size, hidden_size)
        self.act = nn.ELU()

        self.conv = Seq(Conv2d(hidden_size, 2 * hidden_size, 1),
                        nn.ELU(),
                        Conv2d(2 * hidden_size, 1, 1))

    def forward(self, x_pred: Tensor, x_prey: Tensor, x_obst: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        N = (x_pred.size(1), x_prey.size(1), x_obst.size(1))  # type: ignore

        h = torch.cat([x_pred, x_prey, x_obst], dim=1)
        W = (h.unsqueeze(2) - h.unsqueeze(1)).permute(0, 3, 1, 2)
        W = self.conv(W).squeeze()

        h = h + self.act(self.lin(W @ h))

        out_pred, out_prey, out_obst = h.split(N, dim=1)
        return out_pred, out_prey, out_obst
