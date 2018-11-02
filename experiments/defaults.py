import torch
from proj.common.utils import HSeries
import proj.common.models as models

models = {
    'GAEcartpole': (
        dict(
            policy=models.MlpPolicy,
            baseline=models.MlpBaseline,
        ),
        dict(
            policy=dict(hidden_sizes=[]),
            baseline=dict(hidden_sizes=[20]),
        )
    ),
    'GAErobot': (
        dict(
            policy=models.MlpPolicy,
            baseline=models.MlpBaseline,
        ),
        dict(
            policy=dict(hidden_sizes=[100,50,25]),
            baseline=dict(hidden_sizes=[100,50,25]),
        )
    ),
    '10': (
        dict(
            policy=models.MlpPolicy,
            baseline=models.MlpBaseline,
        ),
        dict(
            policy=dict(hidden_sizes=[10], activation=torch.nn.ELU),
            baseline=dict(hidden_sizes=[10], activation=torch.nn.ELU),
        )
    ),
    '32-32': (
        dict(
            policy=models.MlpPolicy,
            baseline=models.MlpBaseline,
        ),
        dict(
            policy=dict(hidden_sizes=[32,32], activation=torch.nn.ELU),
            baseline=dict(hidden_sizes=[32,32], activation=torch.nn.ELU),
        )
    ),
    '32-32-32': (
        dict(
            policy=models.MlpPolicy,
            baseline=models.MlpBaseline,
        ),
        dict(
            policy=dict(hidden_sizes=[32,32,32], activation=torch.nn.ELU),
            baseline=dict(hidden_sizes=[32,32,32], activation=torch.nn.ELU),
        )
    ),
    '64-64': (
        dict(
            policy=models.MlpPolicy,
            baseline=models.MlpBaseline,
        ),
        dict(
            policy=dict(hidden_sizes=[64,64], activation=torch.nn.ELU),
            baseline=dict(hidden_sizes=[64,64], activation=torch.nn.ELU),
        )
    ),
}

optims = {
    'Adam': (
        dict(
            optimizer=torch.optim.Adam,
            scheduler=torch.optim.lr_scheduler.ExponentialLR,
        ),
        dict(
            optimizer=dict(lr=1e-2),
            scheduler=dict(gamma=1.0),
        )
    ),
    'SGD': (
        dict(
            optimizer=torch.optim.SGD,
            scheduler=torch.optim.lr_scheduler.LambdaLR,
        ),
        dict(
            optimizer=dict(lr=1.0),
            scheduler=dict(lr_lambda=HSeries(6)),
            
        )
    )
}

def models_config(model, optim=None):
    global models
    global optims
    types, args = models[model]
    if optim is not None:
        x, y = optims[optim]
        types.update(x)
        args.update(y)
    return types, args
