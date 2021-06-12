from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple, Union
from typing_extensions import Literal

import torch
from torch import Tensor, nn
from torch.nn import Sequential as Seq, Linear as Lin, LeakyReLU as LReLU

from torch_geometric.nn import NNConv

from ..net import FFN
from ..utils import reset


@dataclass
class CriticConfig:
    input_size: Union[int, Tuple[int, int, int]] = 3
    hidden_sizes: Sequence[int] = (64, 64, 64)

    max_grad_norm: float = 0.5

    lr: float = 3e-4

    entity: Optional[Literal['predator', 'prey']] = None


class PredatorCritic(nn.Module):
    def __init__(self, config: CriticConfig):
        super().__init__()
        self.pos_embedding = FFN([3] + [config.hidden_sizes[0]] * 2, layer_norm=False)
        self.entity_embedding = nn.Embedding(3, config.hidden_sizes[0])

        self.edge_type_embedding = nn.Embedding(10, 10)
        self.action_embedding = FFN([1] + [config.hidden_sizes[0]], layer_norm=False)

        self.conv1 = NNConv(
            config.hidden_sizes[0],
            config.hidden_sizes[0],
            nn=FFN([12, config.hidden_sizes[0], config.hidden_sizes[0] ** 2], layer_norm=True),
            aggr='mean'
        )

        self.act = LReLU(0.1)

        self.net = FFN(list(config.hidden_sizes) + [1])

        self.reset_parameters()

    def reset_parameters(self):
        self.pos_embedding.reset_parameters()
        self.action_embedding.reset_parameters()
        nn.init.xavier_normal_(self.edge_type_embedding.weight)
        nn.init.xavier_normal_(self.entity_embedding.weight)
        self.conv1.reset_parameters()
        self.net.reset_parameters()
        self.net.seq[-2].weight.data.uniform_(-1e-2, 1e-2)

    def forward(self, state, action):
        x = state.x
        edge_index = state.edge_index
        row, col = edge_index
        edge_attr = state.edge_attr
        mask = state.mask
        # is_dead_mask = state.is_dead_mask

        pos = x[:, :2].clone()
        rel_coords = pos[row] - pos[col]
        e = torch.cat([self.edge_type_embedding(edge_attr), rel_coords], dim=1)

        x = self.pos_embedding(x) + self.entity_embedding(mask)
        x[mask == 0] += self.action_embedding(action)

        x = x + self.act(self.conv1(x, edge_index, e))

        x = x[mask == 0]
        x = self.net(x + self.action_embedding(action))
        return x


class PreyCritic(nn.Module):
    def __init__(self, config: CriticConfig):
        super().__init__()
        self.pos_embedding = FFN([3] + [config.hidden_sizes[0]] * 2, layer_norm=False)
        self.entity_embedding = nn.Embedding(3, config.hidden_sizes[0])

        self.edge_type_embedding = nn.Embedding(10, 10)
        self.action_embedding = FFN([1] + [config.hidden_sizes[0]], layer_norm=False)

        self.conv1 = NNConv(
            config.hidden_sizes[0],
            config.hidden_sizes[0],
            nn=FFN([12, config.hidden_sizes[0], config.hidden_sizes[0] ** 2], layer_norm=True),
            aggr='mean'
        )

        self.act = LReLU(0.1)

        self.net = FFN(list(config.hidden_sizes) + [1])

        self.reset_parameters()

    def reset_parameters(self):
        self.pos_embedding.reset_parameters()
        self.action_embedding.reset_parameters()
        nn.init.xavier_normal_(self.edge_type_embedding.weight)
        nn.init.xavier_normal_(self.entity_embedding.weight)
        self.conv1.reset_parameters()
        self.net.reset_parameters()
        self.net.seq[-2].weight.data.uniform_(-1e-2, 1e-2)

    def forward(self, state, action):
        x = state.x
        edge_index = state.edge_index
        row, col = edge_index
        edge_attr = state.edge_attr
        mask = state.mask
        # is_dead_mask = state.is_dead_mask

        pos = x[:, :2].clone()
        rel_coords = pos[row] - pos[col]
        e = torch.cat([self.edge_type_embedding(edge_attr), rel_coords], dim=1)

        x = self.pos_embedding(x) + self.entity_embedding(mask)
        x[mask == 1] += self.action_embedding(action)

        x = x + self.act(self.conv1(x, edge_index, e))

        x = x[mask == 1]
        x = self.net(x + self.action_embedding(action))
        return x
