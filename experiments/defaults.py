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
}


def MlpModels(hidden_sizes):
    return (
        dict(
            policy=models.MlpPolicy,
            baseline=models.MlpBaseline,
        ),
        dict(
            policy=dict(hidden_sizes=hidden_sizes, activation=torch.nn.ELU),
            baseline=dict(hidden_sizes=hidden_sizes, activation=torch.nn.ELU),
        )
    )


def SGDconf(decay):
    return (
        dict(
            optimizer=torch.optim.SGD,
            scheduler=torch.optim.lr_scheduler.LambdaLR,
        ),
        dict(
            optimizer=dict(lr=1.0),
            scheduler=dict(lr_lambda=HSeries(decay)),
            
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
        # expect 'Mlp-size-size-...
        model = model.split('-')
        types, args = MlpModels(list(map(int, model[1:])))
    else:
        types, args = (m.copy() for m in GAEmodels[model])

    if optim is not None:
        if 'SGD' in optim:
            optim = optim.split('-')
            x, y = SGDconf(int(optim[-1]))
        elif 'Adam' in optim:
            optim = optim.split('-')
            x, y = Adamconf(float(optim[-1]))
        else:
            raise ValueError("Invalid optimizer argument {}".format(optim))
        types.update(x)
        args.update(y)
    return types, args
