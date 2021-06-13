from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.nn import Module

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
    return optim

class Agent(ABC):
    _device: torch.device = torch.device('cpu')

    @abstractmethod
    def act(self, state):
        pass

    def to(self, device):
        device = torch.device(device)
        for k, v in self.__dict__.items():
            if hasattr(v, 'to'):
                self.__dict__[k] = v.to(device)
            if 'optim' in k:
                self.__dict__[k] = optimizer_to(v, device)
        self._device = device
        return self

    def cpu(self):
        return self.to('cpu')

    def cuda(self, idx=0):
        return self.to(f'cuda:{idx}')

    @property
    def device(self):
        return self._device
