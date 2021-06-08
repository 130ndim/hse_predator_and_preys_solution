from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.nn import Module


class Agent(ABC):
    _device: torch.device = torch.device('cpu')

    @abstractmethod
    def act(self, state):
        pass

    def to(self, device):
        device = torch.device(device)
        for k, v in self.__dict__.items():
            if isinstance(v, (Module, Tensor)):
                self.__dict__[k] = v.to(device)
        self._device = device
        return self

    def cpu(self):
        return self.to('cpu')

    def cuda(self, idx=0):
        return self.to(f'cuda:{idx}')

    @property
    def device(self):
        return self._device
