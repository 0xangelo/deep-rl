import torch
from proj.common import logger
import proj.common.models as models

def make_policy(env):
    return models.MlpPolicy(env, hidden_sizes=[30,30,30], activation=torch.nn.ELU)


def make_baseline(env):
    return models.MlpBaseline(env, hidden_sizes=[10], activation=torch.nn.ELU)


def make_optim(parameters):
    if 'natural' in logger.get_dir():
        optimizer = torch.optim.SGD(parameters, lr=1.0)
        harmonic = lambda epoch: 1/(epoch//6 + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, harmonic)
    else:
        optimizer = torch.optim.SGD(parameters, lr=1.0)
        harmonic = lambda epoch: 1/(epoch//6 + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, harmonic)
        # optimizer = torch.optim.Adam(parameters, lr=1e-2)
        # scheduler = None
    return (optimizer, scheduler)

