from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple, Union
from typing_extensions import Literal


import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .net import FFN
from .net.layers import MHA, Embedding, PointConv, Atan2

import os

print(os.getcwd())
from utils.buffer import State


@dataclass
class ActorConfig:
    input_size: Union[int, Tuple[int, int, int]] = (2, 2, 3)
    hidden_sizes: Sequence[int] = (64, 64, 64)

    max_grad_norm: float = 0.5

    lr: float = 1e-3

    entity: Optional[Literal['predator', 'prey']] = None


class Actor(nn.Module):
    def __init__(self, config: ActorConfig):
        super().__init__()
        self.net = FFN([config.input_size] + list(config.hidden_sizes) + [2])
        self.net.add_module(str(len(self.net)), nn.Tanh())  # type: ignore

    def forward(self, state: Tensor):
        out = self.net(state)
        angle = torch.atan2(*out.T)
        normalized = angle / math.pi
        return normalized


class GNNActor(nn.Module):
    def __init__(self, config: ActorConfig):
        super().__init__()
        self.embedding = Embedding(config.hidden_sizes[0])
        self.mha = MHA(hs=config.hidden_sizes[0])

        self._entity = config.entity

        self.net = FFN(list(config.hidden_sizes) + [1])
        self.net.seq.add_module(str(len(self.net.seq)), nn.Tanh())
        self.entity_idx = int(config.entity == 'prey')

    def forward(self, state: State):
        pred_state, prey_state, obst_state = state.pred_state, state.prey_state, state.obst_state

        h_pred, h_prey, h_obst = self.embedding(pred_state, prey_state, state.obst_state)

        pos = torch.cat([pred_state, prey_state, obst_state[..., :-1]], dim=1)

        out_ = self.mha(h_pred, h_prey, h_obst, pos)

        out = out_[self._entity == 'prey']
        out = self.net(out)

        return out


class PCActor(nn.Module):
    def __init__(self, config: ActorConfig):
        super().__init__()
        self.embedding = nn.Embedding(5, config.hidden_sizes[0])
        self.obst_embedding = nn.Linear(1, config.hidden_sizes[0])

        self.pos_embedding = nn.Linear(2, config.hidden_sizes[0])

        self.conv1 = PointConv(config.hidden_sizes[0])
        self.conv2 = PointConv(config.hidden_sizes[0])

        self._entity = config.entity

        self.net = FFN(list(config.hidden_sizes) + [2])
        self.net.seq.add_module(str(len(self.net.seq)), nn.Tanh())
        self.entity_idx = int(config.entity == 'prey')

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.embedding.weight)
        nn.init.xavier_normal_(self.obst_embedding.weight)
        nn.init.zeros_(self.obst_embedding.bias)
        self.conv1.reset_parameters()
        self.net.reset_parameters()
        self.net.seq[-2].weight.data.uniform_(-1e-2, 1e-2)

    def forward(self, state: State):
        B = state.pred_state.size(0)

        pos_pred = state.pred_state / 9.
        pos_prey, is_alive_prey = state.prey_state.split((2, 1), dim=-1)
        pos_prey = pos_prey / 9.
        is_alive_prey = is_alive_prey.squeeze(-1).long()

        pos_obst, r_obst = state.obst_state.split((2, 1), dim=-1)
        r_obst = (r_obst - 0.8) / 0.7
        pos_obst /= 9.

        # print(is_alive_prey, is_alive_prey.size(), self.embedding(is_alive_prey).size())
        # print(self.embedding.weight[:, 3].repeat(B, pos_prey.size(1), 1).size())
        x_pred = self.embedding.weight[2].repeat((B, pos_pred.size(1), 1)) + self.pos_embedding(pos_pred)
        x_prey = self.embedding.weight[3].repeat((B, pos_prey.size(1), 1)) + self.embedding(is_alive_prey) + self.pos_embedding(pos_prey)
        x_obst = self.embedding.weight[4].repeat((B, pos_obst.size(1), 1)) + self.obst_embedding(r_obst) + self.pos_embedding(pos_obst)

        out = self.conv1(x_pred, x_prey, x_obst, pos_pred, pos_prey, pos_obst)[self.entity_idx]

        # out = self.conv2(x_pred, x_prey, x_obst, x_pred, x_prey, x_obst)[self.entity_idx]

        out = self.net.seq(out)

        out = torch.atan2(*out.permute(2, 0, 1)).unsqueeze(-1)
        out /= math.pi
        return out

