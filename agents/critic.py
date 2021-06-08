from dataclasses import dataclass

import math

from typing import Optional, Sequence, Tuple, Union
from typing_extensions import Literal

import torch
from torch import Tensor, nn

from .net import FFN
from .net.layers import MHA, Embedding

from utils.buffer import State


@dataclass
class CriticConfig:
    input_size: Union[int, Tuple[int, int, int]] = 3
    hidden_sizes: Sequence[int] = (128, 128)

    lr: float = 3e-4

    entity: Optional[Literal['predator', 'prey']] = None


class Critic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = FFN(config.hidden_sizes + [1])

    def forward(self, state: Tensor, action: Tensor):
        out = self.net(torch.cat([state, action], dim=-1))
        return out


class GNNCritic(nn.Module):
    def __init__(self, config: CriticConfig):
        super().__init__()
        self.embedding = Embedding(config.hidden_sizes[0])
        self.mha = MHA(hs=config.hidden_sizes[0])

        self.net = FFN(list(config.hidden_sizes) + [1])
        self.net.seq.add_module(str(len(self.net.seq)), nn.Tanh())  # type: ignore

        self.action_lin = nn.Linear(2, config.hidden_sizes[0])

        self._entity = config.entity

    def forward(self, state: State, action: Tensor):
        pred_state, prey_state, obst_state = state.pred_state, state.prey_state, state.obst_state

        h_pred, h_prey, h_obst = self.embedding(pred_state, prey_state, state.obst_state)
        action *= math.pi
        action_emb = self.action_lin(torch.cat([action.sin(), action.cos()], dim=-1))

        pos = torch.cat([pred_state, prey_state, obst_state[..., :-1]], dim=1)

        if self._entity == 'prey':
            h_prey += action_emb
        else:
            h_pred += action_emb

        out = (h_pred, h_prey, h_obst)
        for i in range(2):
            out = self.mha(*out, pos)

        out = out[self._entity == 'prey']
        out = self.net(out)

        return out
