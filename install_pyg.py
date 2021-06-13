import os.path as osp

import pathlib
import subprocess
import sys

import torch

import importlib

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
suffix = 'cpu'
for library in [
        '_version', '_convert', '_diag', '_spmm', '_spspmm', '_metis', '_rw',
        '_saint', '_sample', '_ego_sample', '_relabel'
]:
    torch.ops.load_library(importlib.machinery.PathFinder().find_spec(
        f'{library}_{suffix}', [osp.dirname(__file__)]).origin)