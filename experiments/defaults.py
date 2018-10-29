import torch
from proj.common.utils import HSeries
import proj.common.models as models

def models_config():
    types = dict(
        policy=models.MlpPolicy,
        baseline=models.MlpBaseline,
        optimizer=torch.optim.SGD,
        scheduler=torch.optim.lr_scheduler.LambdaLR,
    )
    args = dict(
        policy=dict(hidden_sizes=[30, 30, 30], activation=torch.nn.ELU),
        baseline=dict(hidden_sizes=[30, 30, 30], activation=torch.nn.ELU),
        optimizer=dict(lr=1.0),
        scheduler=dict(lr_lambda=HSeries(6)),
    )
    return types, args
