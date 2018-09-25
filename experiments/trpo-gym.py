import os, json, torch, click
from proj.common import logger
from proj.common.env_makers import EnvMaker
from proj.common.saver import SnapshotSaver
from proj.common.utils import set_global_seeds
from proj.common.models import MlpPolicy, MlpBaseline
from proj.common.tqdm_util import tqdm_out
from proj.algorithms import trpo


@click.command()
@click.argument("env")
@click.option("--log_dir", help="where to save checkpoint and progress data", type=str, default='data/')
@click.option("--n_iter", help="number of iterations to run", type=int, default=100)
@click.option("--n_batch", help="number of samples per iter", type=int, default=2000)
@click.option("--n_envs", help="number of environments to run in parallel", type=int, default=8)
@click.option("--delta", help="kl divergence constraint per step", type=float, default=0.01)
@click.option("--kl_sample_frac", help="fraction of samples for kl computation", type=float, default=0.4)
@click.option("--interval", help="interval between each snapshot", type=int, default=10)
@click.option("--seed", help="for repeatability", type=int, default=None)
def main(env, log_dir, n_iter, n_batch, n_envs, delta, kl_sample_frac,
         interval, seed):
    """Runs TRPO on given environment with specified parameters."""
    
    seed = set_global_seeds(seed)
    exp_name = 'trpo-' + env 
    log_dir += exp_name + '-' + str(seed) + '/'
    os.system("rm -rf {}".format(log_dir))

    with tqdm_out(), logger.session(log_dir):
        with open(os.path.join(log_dir, 'variant.json'), 'at') as fp:
            json.dump(dict(exp_name=exp_name, seed=seed), fp)

        env_maker = EnvMaker(env)
        env = env_maker.make()
        ob_space, ac_space = env.observation_space, env.action_space
        policy = MlpPolicy(ob_space, ac_space)
        baseline = MlpBaseline(ob_space, ac_space)

        trpo(
            env=env,
            env_maker=env_maker,
            policy=policy,
            baseline=baseline,
            n_iter=n_iter,
            n_batch=n_batch,
            n_envs=n_envs,
            delta=delta,
            kl_sample_frac=kl_sample_frac,
            snapshot_saver=SnapshotSaver(log_dir, interval=interval)
        )

if __name__ == "__main__":
    main()
