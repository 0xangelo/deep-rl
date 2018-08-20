import os
import torch
from proj.common import logger
from proj.common.utils import SnapshotSaver
from proj.common.env_makers import EnvMaker
from proj.common.models import POLICY, BASELINE
from proj.ALG import ALG
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred import SETTINGS
SETTINGS['CAPTURE_MODE']='no'
ex = Experiment('EXPERIMENT') #Replace with experiment name
ex.observers.append(MongoObserver.create(db_name='pgtorch'))

@ex.config
def config():
    log_dir = 'data/EXPERIMENT/' #Replace with target dir
    n_iter = 0#
    n_batch = 0#
    n_envs = 0#
    lr = 0#

@ex.automain
def main(log_dir, n_iter, n_batch, n_envs, lr, _seed):
    torch.manual_seed(_seed)
    os.system("rm -rf {}".format(log_dir))

    with logger.session(log_dir):
        env_maker = EnvMaker('ENVIRONMENT') #Replace with desired env
        env = env_maker.make()
        ob_space, ac_space = env.observation_space, env.action_space
        policy = POLICY(ob_space, ac_space)
        baseline = BASELINE(ob_space, ac_space)
        optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

        ALG(
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
