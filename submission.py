# import os.path as osp

import pathlib
# import subprocess
import sys

import torch
import torch_geometric

import numpy as np

dirpath = pathlib.Path(__file__).parent
sys.path.append(str(dirpath))


from .utils.pyg_buffer import state2tensor
from .agents.pyg import PredatorActor, PreyActor, ActorConfig

# Эти классы (неявно) имплементируют необходимые классы агентов
# Если Вам здесь нужны какие-то локальные импорты, то их необходимо относительно текущего пакета
# Пример: `import .utils`, где файл `utils.py` лежит рядом с `submission.py`


class OUNoise:
    state = None

    def __init__(self, scale=0.2, mu=0, theta=0.15, sigma=0.2):
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        # self.reset()

    # def reset(self):
    #     self.state = np.ones((self.n_entities, self.action_dim)) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(*x.shape)
        self.state = x + dx
        return self.state * self.scale

    def __call__(self, other: np.ndarray):
        if self.state is None:
            self.state = np.ones_like(other) * self.mu
        return other + self.noise()


class PredatorAgent:
    def __init__(self, path='pred_ddpg_actor.pt'):
        self.actor = PredatorActor(ActorConfig())
        self.actor.load_state_dict(torch.load(path, map_location='cpu'))

        self._noise = OUNoise()

    @torch.no_grad()
    def act(self, state_dict):
        return self._noise(self.actor(state2tensor(state_dict)).cpu().numpy())


class PreyAgent:
    def __init__(self, path='prey_ddpg_actor.pt'):
        self.actor = PreyActor(ActorConfig())
        self.actor.load_state_dict(torch.load(path, map_location='cpu'))

        self._noise = OUNoise()

    def act(self, state_dict):
        return self._noise(self.actor(state2tensor(state_dict)).cpu().numpy())
