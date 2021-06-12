from typing import Optional
from typing_extensions import TypedDict

from collections import deque
from dataclasses import dataclass

import torch
from torch import Tensor, BoolTensor

from torch_geometric.data import Data, Batch


def state2tensor(dict_, normalize=True):
    state = []
    prey_is_alive = []
    # for idx, entity in enumerate(('predators', 'preys')):
    for obj in dict_['predators']:
        state.append([obj['x_pos'], obj['y_pos'], obj['radius']])
    for obj in dict_['preys']:
        state.append([obj['x_pos'], obj['y_pos'], obj['radius']])
        prey_is_alive.append(obj['is_alive'])
    for obj in dict_['obstacles']:
        state.append([obj['x_pos'], obj['y_pos'], obj['radius']])

    state = torch.tensor(state, dtype=torch.float)
    prey_is_alive = torch.tensor(prey_is_alive, dtype=torch.bool)

    pd_size, py_size, ot_size = len(dict_['predators']), len(dict_['preys']), len(dict_['obstacles'])
    is_dead_mask = torch.cat([torch.zeros(pd_size, dtype=torch.bool), ~prey_is_alive, torch.zeros(ot_size, dtype=torch.bool)])
    mask = torch.tensor([0] * pd_size + [1] * py_size + [2] * ot_size, dtype=torch.int)
    E = mask.view(-1, 1) + mask
    E[pd_size:, pd_size:] += 1
    E += 1
    E[is_dead_mask.view(-1, 1) & is_dead_mask] = 0

    edge_index = E.nonzero().T
    edge_attr = E[edge_index.tolist()]

    if normalize:
        state[:, :2] /= 9.
        state[:, 2] = (state[:, 2] - 0.8) / 0.7

    data = Data(x=state, edge_index=edge_index, edge_attr=edge_attr, mask=mask, is_dead_mask=is_dead_mask)
    return data


@dataclass
class PyGBatch:
    state: Batch
    action: Tensor
    next_state: Batch
    reward: Tensor
    done: Tensor

    def __repr__(self):
        repr_ = ',\n    '.join([f'{k}={list(v.size())}' if "state" not in k else f'{k}={v}' for k, v in self.__dict__.items()])
        return f'{self.__class__.__name__}(\n    {repr_}\n)'

    def __iter__(self):
        yield from self.__dict__.values()

    def to(self, device):
        for k, v in self.__dict__.items():
            if hasattr(v, 'to'):
                self.__dict__[k] = v.to(device)
        return self

    def cpu(self):
        return self.to('cpu')

    def cuda(self, idx: int = 0):
        return self.to(f'cuda:{idx}')


EntityConfig = TypedDict('EntityConfig', n=int, state_dim=int, action_dim=Optional[int])  # type: ignore


class FasterBuffer:
    def __init__(self,
                 predator_config: Optional[EntityConfig] = None,
                 prey_config: Optional[EntityConfig] = None,
                 obstacle_config: Optional[EntityConfig] = None,
                 normalize: bool = True,
                 buffer_size: int = 10000):

        self.predator_config = predator_config or {'n': 2, 'state_dim': 2, 'action_dim': 1}
        self.prey_config = prey_config or {'n': 5, 'state_dim': 2, 'action_dim': 1}
        self.obstacle_config = obstacle_config or {'n': 10, 'state_dim': 3}
        self.normalize = normalize
        self.buffer_size = buffer_size

        self._state_buffer = [None] * buffer_size
        self._next_state_buffer = [None] * buffer_size

        self._pred_actions = torch.empty(buffer_size, self.predator_config['n'])
        self._prey_actions = torch.empty(buffer_size, self.prey_config['n'])

        self._pred_rewards = torch.empty(buffer_size, self.predator_config['n'])
        self._prey_rewards = torch.empty(buffer_size, self.prey_config['n'])
        self._dones = torch.empty(buffer_size, 1)

        self._idx = 0
        self._n_transitions = 0

    def append(self, state, action, next_state, reward, done):
        self._idx %= self.buffer_size
        state, next_state = state2tensor(state), state2tensor(next_state)

        self._state_buffer[self._idx] = state
        self._next_state_buffer[self._idx] = next_state

        self._pred_actions[self._idx] = \
            torch.tensor(action['predators'], dtype=torch.float)
        self._prey_actions[self._idx] = \
            torch.tensor(action['preys'], dtype=torch.float)

        self._pred_rewards[self._idx] = torch.tensor(reward['predators'], dtype=torch.float)
        self._prey_rewards[self._idx] = torch.tensor(reward['preys'], dtype=torch.float)

        self._dones[self._idx] = torch.tensor([done], dtype=torch.float)

        self._idx += 1
        self._n_transitions += 1

    def get_batch(self, size, entity):
        assert entity in {'predator', 'prey'}
        idx = torch.randperm(min(self._n_transitions, self.buffer_size))[:size]
        states = Batch.from_data_list([self._state_buffer[i] for i in idx])
        next_states = Batch.from_data_list([self._next_state_buffer[i] for i in idx])

        action = self._pred_actions[idx] if entity == 'predator' else self._prey_actions[idx]

        reward = self._pred_rewards[idx] if entity == 'predator' else self._prey_rewards[idx]

        done = self._dones[idx]
        return PyGBatch(
            states,
            action.view(-1, 1),
            next_states,
            reward.view(-1, reward.size(-1)),
            done
        )


if __name__ == '__main__':
    from predators_and_preys_env.env import PredatorsAndPreysEnv
    import numpy as np

    env = PredatorsAndPreysEnv(render=False)

    buffer = FasterBuffer()

    done = True
    step_count = 0
    for _ in range(200):
        if done:
            print("reset")
            env.reset()
            step_count = 0

        state, reward, _ = env.step(np.zeros(env.predator_action_size), np.ones(env.prey_action_size))
        next_state, reward, done = env.step(np.zeros(env.predator_action_size), np.ones(env.prey_action_size))
        print(state)
        buffer.append(state,
                      {'predators': [0] * env.predator_action_size,
                       'preys': [1] * env.prey_action_size},
                      next_state,
                      {'predators': [0] * env.predator_action_size,
                       'preys': [1] * env.prey_action_size},
                      done)
        step_count += 1

    a, b, c, d, e = buffer.get_batch(64, 'predator')
    # for
    # batch = buffer.get_batch(64, 'predator')
    # print(batch.state.pred_state)
    # print(buffer._buffer[:3])
