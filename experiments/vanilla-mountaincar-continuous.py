import os
import torch
from proj.common import logger
from proj.common.utils import SnapshotSaver
from proj.common.env_makers import EnvMaker
from proj.common.models import MlpPolicy, MlpBaseline
from proj.vanilla import vanilla
from sacred import Experiment
ex = Experiment('vanilla-mountaincar-continuous')

@ex.config
def config():
    log_dir = 'data/vanilla-mountaicar-continuous/'
    n_iter = 150
    n_batch = 8000
    n_envs = 16
    lr = 1e-3

@ex.automain
def main(log_dir, n_iter, n_batch, n_envs, lr, _seed):
    torch.manual_seed(_seed)
    os.system("rm -rf {}".format(log_dir))

    with logger.session(log_dir):
        logger.set_level(logger.INFO)
        env_maker = EnvMaker('MountainCarContinuous-v0')
        env = env_maker.make()
        ob_space, ac_space = env.observation_space, env.action_space
        policy = MlpPolicy(ob_space, ac_space)
        baseline = MlpBaseline(ob_space, ac_space)
        optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

        vanilla(
            env=env,
            env_maker=env_maker,
            policy=policy,
            baseline=baseline,
            n_iter=n_iter,
            n_batch=n_batch,
            n_envs=n_envs,
            optimizer=optimizer,
            snapshot_saver=SnapshotSaver(log_dir)
        )
