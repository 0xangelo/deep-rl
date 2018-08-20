import os
import torch
from proj.common import logger
from proj.common.utils import SnapshotSaver
from proj.common.env_makers import EnvMaker
from proj.common.models import MlpPolicy, MlpBaseline
from proj.vanilla import vanilla
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred import SETTINGS
SETTINGS['CAPTURE_MODE']='no'
ex = Experiment('vanilla-cartpole')
ex.observers.append(MongoObserver.create(db_name='pgtorch'))

@ex.config
def config():
    log_dir = 'data/vanilla-cartpole/' 
    n_iter = 100
    n_batch = 2000
    n_envs = 4
    lr = 1e-3

@ex.automain
def main(log_dir, n_iter, n_batch, n_envs, lr, _seed):
    torch.manual_seed(_seed)
    os.system("rm -rf {}".format(log_dir))

    with logger.session(log_dir):
        env_maker = EnvMaker('CartPole-v0')
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
