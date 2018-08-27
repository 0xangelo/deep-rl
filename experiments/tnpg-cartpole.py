import os
import torch
from proj.common import logger
from proj.common.utils import SnapshotSaver
from proj.common.env_makers import EnvMaker
from proj.common.models import MlpPolicy, MlpBaseline
from proj.common.tqdm_util import tqdm_out
from proj.algorithms import tnpg
from sacred import SETTINGS, Experiment
from sacred.observers import MongoObserver
SETTINGS['CAPTURE_MODE']='no'
ex = Experiment('tnpg-cartpole')
ex.observers.append(MongoObserver.create(db_name='pgtorch'))

@ex.config
def config():
    log_dir = 'data/'
    n_iter = 100
    n_batch = 2000
    n_envs = 4
    step_size = 0.01
    kl_subsamp_ratio = 0.8
    interval = 1

@ex.automain
def main(log_dir, n_iter, n_batch, n_envs, step_size, kl_subsamp_ratio, interval, seed):
    torch.manual_seed(seed)
    log_dir += ex.path + '-' + str(seed) + '/'
    os.system("rm -rf {}".format(log_dir))

    with tqdm_out(), logger.session(log_dir):
        env_maker = EnvMaker('CartPole-v0')
        env = env_maker.make()
        ob_space, ac_space = env.observation_space, env.action_space
        policy = MlpPolicy(ob_space, ac_space)
        baseline = MlpBaseline(ob_space, ac_space)

        tnpg(
            env=env,
            env_maker=env_maker,
            policy=policy,
            baseline=baseline,
            n_iter=n_iter,
            n_batch=n_batch,
            n_envs=n_envs,
            step_size=step_size,
            kl_subsamp_ratio=kl_subsamp_ratio,
            snapshot_saver=SnapshotSaver(log_dir, interval=interval)
        )
