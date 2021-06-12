import pathlib

import sys
import os.path as osp

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
