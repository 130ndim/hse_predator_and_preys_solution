from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple, Union
from typing_extensions import Literal


import torch
from torch import Tensor, nn

from .net import FFN
from .net.layers import MHA, Embedding

import os

print(os.getcwd())
from utils.buffer import State


@dataclass
class ActorConfig:
    input_size: Union[int, Tuple[int, int, int]] = (2, 2, 3)
    hidden_sizes: Sequence[int] = (128, 128)

    lr: float = 3e-4

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

        self.net = FFN(list(config.hidden_sizes) + [1])
        self.net.seq.add_module(str(len(self.net.seq)), nn.Tanh())  # type: ignore

        self.action_lin = nn.Linear(2, config.hidden_sizes[0])

        self._entity = config.entity

        self.net = FFN(list(config.hidden_sizes) + [2])
        self.net.seq.add_module(str(len(self.net.seq)), nn.Tanh())
        self.entity_idx = int(config.entity == 'prey')

    def forward(self, state: State):
        pred_state, prey_state, obst_state = state.pred_state, state.prey_state, state.obst_state

        h_pred, h_prey, h_obst = self.embedding(pred_state, prey_state, state.obst_state)

        pos = torch.cat([pred_state, prey_state, obst_state[..., :-1]], dim=1)

        out = (h_pred, h_prey, h_obst)
        for i in range(2):
            out = self.mha(*out, pos)

        out = out[self._entity == 'prey']
        out = self.net(out)

        angle = torch.atan2(*out.permute(2, 0, 1))
        normalized = angle / math.pi
        return normalized.unsqueeze(-1)
