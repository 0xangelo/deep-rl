import os
import torch
from proj.common import logger
from proj.common.utils import SnapshotSaver
from proj.common.env_makers import EnvMaker
from proj.common.models import MlpPolicy, MlpBaseline
from proj.common.tqdm_out import redirect
from proj.algorithms import vanilla
from sacred import SETTINGS, Experiment
from sacred.observers import MongoObserver
SETTINGS['CAPTURE_MODE']='no'
ex = Experiment('vanilla-lunarlander')
ex.observers.append(MongoObserver.create(db_name='pgtorch'))

@ex.config
def config():
    log_dir = 'data/'
    n_iter = 250
    n_batch = 8000
    n_envs = 16
    lr = 1e-3

@ex.automain
def main(log_dir, n_iter, n_batch, n_envs, lr, seed):
    torch.manual_seed(seed)
    log_dir += ex.path + '-' + str(seed) + '/'
    os.system("rm -rf {}".format(log_dir))

    with logger.session(log_dir), redirect():
        logger.set_level(logger.INFO)
        env_maker = EnvMaker('LunarLander-v2')
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
