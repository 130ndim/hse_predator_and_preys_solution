from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor

from torch_geometric.data import Data, Batch


class Buffer(object):
    def __init__(self, state_dim: int, action_dim: int, max_size: int):
        self.max_size = max_size

        self.state_buff = [None] * max_size
        self.action_buff = torch.zeros(max_size, action_dim, dtype=torch.float)
        self.reward_buff = torch.zeros(max_size, dtype=torch.float)
        self.next_state_buff = [None] * max_size
        self.done_buff = torch.zeros(max_size, dtype=torch.float)

        self.filled_i = 0
        self.curr_size = 0

    def push(self, state: Data, action: np.ndarray,
             next_state: Data, reward: float, done: bool):
        self.state_buff[self.filled_i] = state
        self.action_buff[self.filled_i] = torch.Tensor(action)
        self.reward_buff[self.filled_i] = torch.Tensor([reward])
        self.next_state_buff[self.filled_i] = next_state
        self.done_buff[self.filled_i] = torch.Tensor([done])

        self.curr_size = min(self.max_size, self.curr_size + 1)
        self.filled_i = (self.filled_i + 1) % self.max_size

    def sample(self, batch_size: int, norm_rew: bool = True) -> Tuple[Batch, Tensor, Batch, Tensor, Tensor]:
        indices = np.random.choice(self.curr_size, batch_size, replace=False)
        indices = torch.Tensor(indices).long()

        if norm_rew:
            mean = torch.mean(self.reward_buff[:self.curr_size])
            std = torch.std(self.reward_buff[:self.curr_size])
            rew = (self.reward_buff[indices] - mean) / std
        else:
            rew = self.reward_buff[indices]

        state = Batch.from_data_list([self.state_buff[i] for i in indices])
        next_state = Batch.from_data_list([self.next_state_buff[i] for i in indices])

        return state, self.action_buff[indices], next_state, rew,  self.done_buff[indices]

    def __len__(self):
        return self.curr_size
