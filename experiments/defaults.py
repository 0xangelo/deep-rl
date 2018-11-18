import torch
from proj.common.utils import HSeries
import proj.common.models as models

GAEmodels = {
    'GAEcartpole': (
        dict(
            policy=models.MlpPolicy,
            baseline=models.MlpBaseline,
        ),
        dict(
            policy=dict(hidden_sizes=[]),
            baseline=dict(hidden_sizes=[20], activation='tanh'),
        )
    ),
    'GAErobot': (
        dict(
            policy=models.MlpPolicy,
            baseline=models.MlpBaseline,
        ),
        dict(
            policy=dict(hidden_sizes=[100,50,25], activation='tanh'),
            baseline=dict(hidden_sizes=[100,50,25], activation='tanh'),
        )
    ),
}


def MlpModels(hidden_sizes, activation):
    return (
        dict(
            policy=models.MlpPolicy,
            baseline=models.MlpBaseline,
        ),
        dict(
            policy=dict(hidden_sizes=hidden_sizes, activation=activation),
            baseline=dict(hidden_sizes=hidden_sizes, activation=activation),
        )
    )


def SGDconf(scale):
    return (
        dict(
            optimizer=torch.optim.SGD,
            scheduler=torch.optim.lr_scheduler.LambdaLR,
        ),
        dict(
            optimizer=dict(lr=1.0),
            scheduler=dict(lr_lambda=HSeries(scale)),
            
        )
    )


def Adamconf(lr):
    return (
        dict(
            optimizer=torch.optim.Adam,
            scheduler=torch.optim.lr_scheduler.ExponentialLR,
        ),
        dict(
            optimizer=dict(lr=lr),
            scheduler=dict(gamma=1.0),
        )
    )


def models_config(model, optim=None):
    if 'Mlp' in model:
        # expect 'Mlp:size-...-size:activation
        model = model.split(':')
        sizes = list(map(int, model[1].split('-')))
        activation = model[2]
        types, args = MlpModels(sizes, activation)
    else:
        types, args = (m.copy() for m in GAEmodels[model])

    if optim is not None:
        optim = optim.split(':')
        if 'SGD' in optim:
            x, y = SGDconf(float(optim[1]))
        elif 'Adam' in optim:
            x, y = Adamconf(float(optim[1]))
        else:
            raise ValueError("Invalid optimizer argument {}".format(optim))
        types.update(x)
        args.update(y)
    return types, args
