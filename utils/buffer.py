from typing import Optional
from typing_extensions import TypedDict

from dataclasses import dataclass

import torch
from torch import Tensor


def dict2state(dict_):
    state = []
    for idx, entity in enumerate(('predators', 'preys')):
        for obj in dict_[entity]:
            state += [obj['x_pos'], obj['y_pos']]
    for obj in dict_['obstacles']:
        state += [obj['x_pos'], obj['y_pos'], obj['radius']]
    return state


class ToDevice:
    def to(self, device):
        for k, v in self.__dict__.items():
            if hasattr(v, 'to'):
                self.__dict__[k] = v.to(device)
        return self

    def cpu(self):
        return self.to('cpu')

    def cuda(self, idx: int = 0):
        return self.to(f'cuda:{idx}')


@dataclass
class State(ToDevice):
    pred_state: Tensor
    prey_state: Tensor
    obst_state: Tensor

    def __repr__(self):
        return f'{self.__class__.__name__}(pred_state={list(self.pred_state.size())}, ' \
               f'prey_state={list(self.prey_state.size())}, obst_state={list(self.obst_state.size())})'


@dataclass
class Batch(ToDevice):
    state: State
    action: Tensor
    next_state: State
    reward: Tensor
    done: Tensor

    @classmethod
    def from_tensors(cls, tensors):
        return cls(State(*tensors[:3]),
                   tensors[3],
                   State(*tensors[4:7]),
                   tensors[7],
                   tensors[8])

    def __repr__(self):
        repr_ = ',\n    '.join([f'{k}={v}' if k in {"state", "next_state"} else f'{k}={list(self.done.size())}' for k, v in self.__dict__.items()])
        return f'{self.__class__.__name__}(\n    {repr_}\n)'

    def __iter__(self):
        for attr in ('state', 'action', 'next_state', 'reward', 'done'):
            yield getattr(self, attr)


EntityConfig = TypedDict('EntityConfig', n=int, state_dim=int)  # type: ignore


class Buffer:
    def __init__(self,
                 predator_config: Optional[EntityConfig] = None,
                 prey_config: Optional[EntityConfig] = None,
                 obstacle_config: Optional[EntityConfig] = None,
                 buffer_size: int = 1000000):
        self.predator_config = predator_config or {'n': 2, 'state_dim': 2}
        self.prey_config = prey_config or {'n': 5, 'state_dim': 2}
        self.obstacle_config = obstacle_config or {'n': 10, 'state_dim': 3}

        self.buffer_size = buffer_size
        # n * (state + next_state + action + reward)
        pred_dim = \
            self.predator_config['n'] * (2 * self.predator_config['state_dim'] + 2)
        prey_dim = self.prey_config['n'] * (2 * self.prey_config['state_dim'] + 2)
        # n * (state + next_state)
        obst_dim = self.obstacle_config['n'] * 2 * self.obstacle_config['state_dim']

        self._buffer = torch.empty(
            buffer_size,
            pred_dim + prey_dim + obst_dim + 1
        )
        self._split_indices = (self.predator_config['n'] * self.predator_config['state_dim'],
                               self.prey_config['n'] * self.prey_config['state_dim'],
                               self.obstacle_config['n'] * self.obstacle_config['state_dim'],
                               self.predator_config['n'],
                               self.prey_config['n'],
                               self.predator_config['n'] * self.predator_config['state_dim'],
                               self.prey_config['n'] * self.prey_config['state_dim'],
                               self.obstacle_config['n'] * self.obstacle_config['state_dim'],
                               self.predator_config['n'],
                               self.prey_config['n'],
                               1)

        self._idx = 0
        self._n_transitions = 0

    def append(self, state, action, next_state, reward, done):
        self._idx %= self.buffer_size
        self._buffer[self._idx] = \
            torch.tensor(dict2state(state) + action['predator_actions'] + action['prey_actions'] +
                         dict2state(next_state) + reward['predators'].tolist() + reward['preys'].tolist() +
                         [done])

        self._idx += 1
        self._n_transitions += 1

    def __getitem__(self, idx):
        out = self._buffer[idx]
        return out

    def get_batch(self, size, entity):
        idx = torch.randperm(min(self._n_transitions, self._buffer.size(0)))[:size]
        out = self[idx]
        batch = []
        for i, t in enumerate(out.split(self._split_indices, dim=1)):
            if i in {0, 5}:
                t = t.view(-1, self.predator_config['n'], self.predator_config['state_dim'])
            elif i in {1, 6}:
                t = t.view(-1, self.prey_config['n'], self.prey_config['state_dim'])
            elif i in {2, 7}:
                t = t.view(-1, self.obstacle_config['n'], self.obstacle_config['state_dim'])
            elif i in {3, 8}:
                t = t.view(-1, self.predator_config['n'], 1)
            elif i in {4, 9}:
                t = t.view(-1, self.prey_config['n'], 1)
            else:
                t = t.view(-1)
            batch.append(t)
        if entity == 'predator':
            batch.pop(9)
            batch.pop(4)
        elif entity == 'prey':
            batch.pop(8)
            batch.pop(3)
        return Batch.from_tensors(batch)


if __name__ == '__main__':
    from predators_and_preys_env.env import PredatorsAndPreysEnv
    import numpy as np

    env = PredatorsAndPreysEnv(render=True)

    buffer = Buffer()
    print(buffer._buffer.size())

    done = True
    step_count = 0
    for _ in range(2):
        if done:
            print("reset")
            env.reset()
            step_count = 0

        state, _ = env.step(np.zeros(env.predator_action_size), np.ones(env.prey_action_size))
        next_state, done = env.step(np.zeros(env.predator_action_size), np.ones(env.prey_action_size))
        buffer.append(dict2state(state),
                      {'predator_actions': [0] * env.predator_action_size,
                       'prey_actions': [1] * env.prey_action_size},
                      dict2state(next_state),
                      {'predator_rewards': [0] * env.predator_action_size,
                       'prey_rewards': [1] * env.prey_action_size},
                      done)
        step_count += 1
    batch = buffer.get_batch(2)
    print(batch)
    print(batch.state.pred_state)
    print(buffer._buffer[:3])
    a, b, c, d, e = batch
    print(a)
    print(b)
    print(c)
    print(d)
    print(e)

