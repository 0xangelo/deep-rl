import os, click, torch
from proj.algorithms import trpo
from proj.common.utils import set_global_seeds
from proj.common.models import *
from proj.common import logger
from proj.common.env_makers import EnvMaker
from proj.common.saver import SnapshotSaver


@click.command()
@click.argument("env")
@click.option("--exp_name", help="group results under this name",
              type=str, default='test')
@click.option("--log_dir", help="where to save checkpoint and progress data",
              type=str, default='data/')
@click.option("--cuda", help="enable GPU acceleration if available",
              is_flag=True)
@click.option("--interval", help="interval between each snapshot",
              type=int, default=10)
@click.option("--seed", help="for repeatability",
              type=int, default=None)
@click.option("--policy", help="policy class to use",
              type=str, default='MlpPolicy')
@click.option("--pol_kwargs", help="policy kwargs",
              type=str, default='dict()')
@click.option("--baseline", help="baseline class to use",
              type=str, default='MlpBaseline')
@click.option("--base_kwargs", help="baseline kwargs",
              type=str, default='dict()')
@click.option("--n_iter", help="number of iterations to run",
              type=int, default=100)
@click.option("--n_batch", help="number of samples per iterations",
              type=int, default=4000)
@click.option("--n_envs", help="number of environments to run in parallel",
              type=int, default=8)
@click.option("--gamma", help="generalized advantage estimation discount",
              type=float, default=0.99)
@click.option("--gaelam", help="generalized advantage estimation lambda",
              type=float, default=0.97)
@click.option("--kl_frac", help="fraction of samples for fisher vector product",
              type=float, default=0.2)
@click.option("--delta", help="kl divergence constraint per step",
              type=float, default=0.01)
def main(env, exp_name, cuda, log_dir, interval, seed,
         policy, baseline, **algargs):
    """Runs TRPO on given environment with specified parameters."""
    
    seed = set_global_seeds(seed)
    if cuda and torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.FloatTensor)
    log_dir = os.path.join(log_dir, exp_name, exp_name + '_' + str(seed), '')
    params = dict(
        exp_name=exp_name,
        alg='trpo',
        env=env,
        seed=seed,
        **algargs
    )
    
    os.system("rm -rf {}".format(log_dir))
    with logger.session(log_dir):
        algargs['env_maker'] = env_maker = EnvMaker(params['env'])
        saver = SnapshotSaver(log_dir, interval=interval)
        saver.save_config(params)
        env = env_maker.make()
        policy = eval(policy)(env, **eval(algargs.pop("pol_kwargs")))
        baseline = eval(baseline)(env, **eval(algargs.pop("base_kwargs")))

        trpo(
            env=env,
            policy=policy,
            baseline=baseline,
            snapshot_saver=saver,
            **algargs
        )


if __name__ == "__main__":
    main()
