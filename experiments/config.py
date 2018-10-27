import torch
from proj.common import logger
from proj.common.utils import HSeries
import proj.common.models as models

def make_policy(env):
    return models.MlpPolicy(env, hidden_sizes=[30,30,30], activation=torch.nn.ELU)


def make_baseline(env):
    return models.MlpBaseline(env, hidden_sizes=[10], activation=torch.nn.ELU)


def make_optim(module):
    if 'natural' in logger.get_dir():
        optimizer = torch.optim.SGD(module.parameters(), lr=1.0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, HSeries(6))
    else:
        optimizer = torch.optim.SGD(module.parameters(), lr=1.0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, HSeries(6))
    return (optimizer, scheduler)

