from typing import Optional
from typing_extensions import TypedDict

from dataclasses import dataclass

import torch
from torch import Tensor, BoolTensor


def state2tensor(dict_, normalize=True):
    pred_state, prey_state, obst_state = [], [], []
    prey_is_alive = []
    # for idx, entity in enumerate(('predators', 'preys')):
    for obj in dict_['predators']:
        pred_state.append([obj['x_pos'], obj['y_pos']])
    for obj in dict_['preys']:
        prey_state.append([obj['x_pos'], obj['y_pos']])
        prey_is_alive.append(obj['is_alive'])
    for obj in dict_['obstacles']:
        obst_state.append([obj['x_pos'], obj['y_pos'], obj['radius']])
    pred_state = torch.tensor(pred_state, dtype=torch.float)
    prey_state = torch.tensor(prey_state, dtype=torch.float)
    prey_is_alive = torch.tensor(prey_is_alive, dtype=torch.bool)
    obst_state = torch.tensor(obst_state, dtype=torch.float)
    if normalize:
        pred_state /= 9.
        prey_state /= 9.
        obst_state[:, :2] /= 9.
        obst_state[:, 2] = (obst_state[:, 2] - 0.8) / 0.7
    return pred_state, prey_state, obst_state, prey_is_alive


@dataclass
class Batch:
    pred_state: Tensor
    prey_state: Tensor
    prey_is_alive_state: BoolTensor
    obst_state: Tensor
    action: Tensor
    pred_next_state: Tensor
    prey_next_state: Tensor
    prey_is_alive_next_state: BoolTensor
    obst_next_state: Tensor
    reward: Tensor
    done: Tensor

    def __repr__(self):
        repr_ = ',\n    '.join([f'{k}={list(v.size())}' for k, v in self.__dict__.items()])
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

        self._pred_states = torch.empty(buffer_size, self.predator_config['n'], self.predator_config['state_dim'])
        self._prey_states = torch.empty(buffer_size, self.prey_config['n'], self.prey_config['state_dim'])
        self._prey_is_alive_states = torch.empty(buffer_size, self.prey_config['n'], dtype=torch.bool)
        self._obst_states = torch.empty(buffer_size, self.obstacle_config['n'], self.obstacle_config['state_dim'])

        self._pred_actions = torch.empty(buffer_size, self.predator_config['n'], self.predator_config['action_dim'])
        self._prey_actions = torch.empty(buffer_size, self.prey_config['n'], self.prey_config['action_dim'])

        self._pred_next_states = torch.empty(buffer_size, self.predator_config['n'], self.predator_config['state_dim'])
        self._prey_next_states = torch.empty(buffer_size, self.prey_config['n'], self.prey_config['state_dim'])
        self._prey_is_alive_next_states = torch.empty(buffer_size, self.prey_config['n'], dtype=torch.bool)
        self._obst_next_states = torch.empty(buffer_size, self.obstacle_config['n'], self.obstacle_config['state_dim'])

        self._pred_rewards = torch.empty(buffer_size, self.predator_config['n'])
        self._prey_rewards = torch.empty(buffer_size, self.prey_config['n'])
        self._dones = torch.empty(buffer_size, 1)

        self._idx = 0
        self._n_transitions = 0

    def append(self, state, action, next_state, reward, done):
        self._idx %= self.buffer_size
        (pred_state, prey_state, obst_state, prey_is_alive_state), \
        (pred_next_state, prey_next_state, obst_next_state, prey_is_alive_next_state) = \
            state2tensor(state), state2tensor(next_state)

        self._pred_states[self._idx] = pred_state
        self._prey_states[self._idx] = prey_state
        self._prey_is_alive_states[self._idx] = prey_is_alive_state
        self._obst_states[self._idx] = obst_state

        self._pred_actions[self._idx] = \
            torch.tensor(action['predators'], dtype=torch.float).view(-1, self.predator_config['action_dim'])
        self._prey_actions[self._idx] = \
            torch.tensor(action['preys'], dtype=torch.float).view(-1, self.prey_config['action_dim'])

        self._pred_next_states[self._idx] = pred_next_state
        self._prey_next_states[self._idx] = prey_next_state
        self._prey_is_alive_next_states[self._idx] = prey_is_alive_next_state
        self._obst_next_states[self._idx] = obst_next_state

        self._pred_rewards[self._idx] = torch.tensor(reward['predators'], dtype=torch.float)
        self._prey_rewards[self._idx] = torch.tensor(reward['preys'], dtype=torch.float)

        self._dones[self._idx] = torch.tensor([done], dtype=torch.float)

        self._idx += 1
        self._n_transitions += 1

    def get_batch(self, size, entity):
        assert entity in {'predator', 'prey'}
        idx = torch.randperm(min(self._n_transitions, self.buffer_size))[:size]
        pred_state, prey_state, obst_state = \
            self._pred_states[idx], self._prey_states[idx], self._obst_states[idx]
        pred_next_state, prey_next_state, obst_next_state = \
            self._pred_states[idx], self._prey_states[idx], self._obst_states[idx]
        prey_is_alive_state, prey_is_alive_next_state = \
            self._prey_is_alive_states[idx], self._prey_is_alive_next_states[idx]

        action = self._pred_actions[idx] if entity == 'predator' else self._prey_actions[idx]

        reward = self._pred_rewards[idx] if entity == 'predator' else self._prey_rewards[idx]

        done = self._dones[idx]
        return Batch(
            pred_state, prey_state, prey_is_alive_state, obst_state,
            action,
            pred_next_state, prey_next_state, prey_is_alive_next_state, obst_next_state,
            reward,
            done
        )


if __name__ == '__main__':
    from predators_and_preys_env.env import PredatorsAndPreysEnv
    import numpy as np

    env = PredatorsAndPreysEnv(render=False)

    buffer = FasterBuffer()

    done = True
    step_count = 0
    for _ in range(2):
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
    batch = buffer.get_batch(2, 'predator')
    print(batch)
    # print(batch.state.pred_state)
    # print(buffer._buffer[:3])
