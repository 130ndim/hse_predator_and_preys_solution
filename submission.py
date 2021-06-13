import os.path as osp

import pathlib
import subprocess
import sys

import torch

torch_version = torch.__version__.split('.')
torch_version[-1] = '0'
torch_version = '.'.join(torch_version)

subprocess.check_call([sys.executable, "-m", "pip", "install", "torch-scatter",
                       "-f", f"https://pytorch-geometric.com/whl/torch-${torch_version}+cpu.html"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch-sparse",
                       "-f", f"https://pytorch-geometric.com/whl/torch-${torch_version}+cpu.html"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch-cluster",
                       "-f", f"https://pytorch-geometric.com/whl/torch-${torch_version}+cpu.html"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch-geometric"])

dirpath = pathlib.Path(__file__).parent
sys.path.append(str(dirpath))


from .agents.td3 import TD3Agent

# Эти классы (неявно) имплементируют необходимые классы агентов
# Если Вам здесь нужны какие-то локальные импорты, то их необходимо относительно текущего пакета
# Пример: `import .utils`, где файл `utils.py` лежит рядом с `submission.py`


class PredatorAgent:
    def __init__(self, path='td3_pred_190000.pt'):
        self.actor = TD3Agent.from_ckpt(osp.join(dirpath, path), 'cpu')

    def act(self, state_dict):
        return self.actor.act(state_dict)


class PreyAgent:
    def __init__(self, path='td3_prey_190000.pt'):
        self.actor = TD3Agent.from_ckpt(osp.join(dirpath, path), 'cpu')

    def act(self, state_dict):
        return self.actor.act(state_dict)
