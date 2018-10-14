import torch
import proj.common.models as models
import proj.common.env_pool as env_pool

env_pool.BOOTSTRAP = False


def make_policy(env):
    return models.MlpPolicy(env)


def make_baseline(env):
    return models.MlpBaseline(env)


def make_optim(parameters):
    optimizer = torch.optim.SGD(parameters, lr=1.0)
    harmonic = lambda epoch: 1/(epoch//3 + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, harmonic)
    return (optimizer, scheduler)

