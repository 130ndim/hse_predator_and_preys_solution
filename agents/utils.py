import numpy as np

import torch
from torch.nn import Module


def soft_update(target: Module, source: Module, tau: float) -> None:
    for p_t, p_s in zip(target.parameters(), source.parameters()):
        p_t.data = tau * p_t.data + (1 - tau) * p_s.data


def hard_update(target: Module, source: Module) -> None:
    for p_t, p_s in zip(target.parameters(), source.parameters()):
        p_t.data.copy_(p_s.data)


# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:
    state = None

    def __init__(self, action_dim, n_entities, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.n_entities = n_entities
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones((self.n_entities, self.action_dim)) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale

    def __add__(self, other: torch.Tensor):
        return other + self.noise


class ZeroCenteredNoise:
    def __init__(self, sigma=0.3, decay=0.999, freq=100):
        self.sigma = sigma
        self.decay = decay
        self.freq = freq

        self._step = 0

    def __call__(self, other):
        out = other + np.random.randn(*other.shape) * self.sigma

        if self._step % self.freq == 0:
            self.sigma *= self.decay
        self._step += 1

        return out
