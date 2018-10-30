import torch
from proj.common.utils import HSeries
import proj.common.models as models

configs = {
    'simple': (
        dict(
            policy=models.MlpPolicy,
            baseline=models.MlpBaseline,
            optimizer=torch.optim.SGD,
            scheduler=torch.optim.lr_scheduler.LambdaLR,
        ),
        dict(
            policy=dict(hidden_sizes=[]),
            baseline=dict(hidden_sizes=[10]),
            optimizer=dict(lr=1.0),
            scheduler=dict(lr_lambda=HSeries(1)),
        )
    ),
    '10SGD': (
        dict(
            policy=models.MlpPolicy,
            baseline=models.MlpBaseline,
            optimizer=torch.optim.SGD,
            scheduler=torch.optim.lr_scheduler.LambdaLR,
        ),
        dict(
            policy=dict(hidden_sizes=[10]),
            baseline=dict(hidden_sizes=[10]),
            optimizer=dict(lr=1.0),
            scheduler=dict(lr_lambda=HSeries(6)),
        )
    ),
    '10Adam': (
        dict(
            policy=models.MlpPolicy,
            baseline=models.MlpBaseline,
            optimizer=torch.optim.Adam,
            scheduler=torch.optim.lr_scheduler.ExponentialLR,
        ),
        dict(
            policy=dict(hidden_sizes=[10]),
            baseline=dict(hidden_sizes=[10]),
            optimizer=dict(lr=1e-2),
            scheduler=dict(gamma=1.0),
        )
    ),
    '32-32SGD': (
        dict(
            policy=models.MlpPolicy,
            baseline=models.MlpBaseline,
            optimizer=torch.optim.SGD,
            scheduler=torch.optim.lr_scheduler.LambdaLR,
        ),
        dict(
            policy=dict(hidden_sizes=[32,32], activation=torch.nn.ELU),
            baseline=dict(hidden_sizes=[32,32], activation=torch.nn.ELU),
            optimizer=dict(lr=1.0),
            scheduler=dict(lr_lambda=HSeries(6)),
        )
    ),
    '32-32Adam': (
        dict(
            policy=models.MlpPolicy,
            baseline=models.MlpBaseline,
            optimizer=torch.optim.Adam,
            scheduler=torch.optim.lr_scheduler.ExponentialLR,
        ),
        dict(
            policy=dict(hidden_sizes=[32,32], activation=torch.nn.ELU),
            baseline=dict(hidden_sizes=[32,32], activation=torch.nn.ELU),
            optimizer=dict(lr=1e-2),
            scheduler=dict(gamma=1.0),
        )
    ),
    '32-32-32SGD': (
        dict(
            policy=models.MlpPolicy,
            baseline=models.MlpBaseline,
            optimizer=torch.optim.SGD,
            scheduler=torch.optim.lr_scheduler.LambdaLR,
        ),
        dict(
            policy=dict(hidden_sizes=[32,32,32], activation=torch.nn.ELU),
            baseline=dict(hidden_sizes=[32,32,32], activation=torch.nn.ELU),
            optimizer=dict(lr=1.0),
            scheduler=dict(lr_lambda=HSeries(6)),
        )
    ),
    '64-64SGD': (
        dict(
            policy=models.MlpPolicy,
            baseline=models.MlpBaseline,
            optimizer=torch.optim.SGD,
            scheduler=torch.optim.lr_scheduler.LambdaLR,
        ),
        dict(
            policy=dict(hidden_sizes=[64,64], activation=torch.nn.ELU),
            baseline=dict(hidden_sizes=[64,64], activation=torch.nn.ELU),
            optimizer=dict(lr=1.0),
            scheduler=dict(lr_lambda=HSeries(6)),
        )
    ),
    '64-64Adam': (
        dict(
            policy=models.MlpPolicy,
            baseline=models.MlpBaseline,
            optimizer=torch.optim.Adam,
            scheduler=torch.optim.lr_scheduler.ExponentialLR,
        ),
        dict(
            policy=dict(hidden_sizes=[64,64], activation=torch.nn.ELU),
            baseline=dict(hidden_sizes=[64,64], activation=torch.nn.ELU),
            optimizer=dict(lr=1e-2),
            scheduler=dict(gamma=1.0),
        )
    )
}

def models_config(index):
    global configs
    return configs[index]
