import click
from proj.algorithms import vanilla, natural, trpo, train
from proj.common.utils import set_global_seeds
from proj.common.env_makers import EnvMaker
from defaults import models_config

@click.command()
@click.argument("env")
@click.option("--log_dir", help="where to save checkpoint and progress data",
              type=str, default='data/')
@click.option("--episodic", help="enforce all episodes end",
              is_flag=True)
@click.option("--n_iter", help="number of iterations to run",
              type=int, default=100)
@click.option("--n_batch", help="number of samples per iterations",
              type=int, default=2000)
@click.option("--n_envs", help="number of environments to run in parallel",
              type=int, default=8)
@click.option("--gamma", help="discount factor for expected return criterion",
              type=float, default=0.99)
@click.option("--gae_lambda", help="generalized advantage estimation factor",
              type=float, default=0.97)
@click.option("--interval", help="interval between each snapshot",
              type=int, default=10)
@click.option("--seed", help="for repeatability",
              type=int, default=None)
@click.option("--delta", help="kl divergence constraint per step",
              type=float, default=1e-3)
@click.option("--kl_frac", help="fraction of samples for kl computation",
              type=float, default=0.4)
def main(env, log_dir, episodic, n_iter, n_batch, n_envs, gamma, gae_lambda,
         interval, seed, delta, kl_frac):
    """
    Runs the algorithms on given environment with specified parameters.
    """
    seed = set_global_seeds(seed)
    proto_dir = log_dir + env + '/{alg}/{seed}/'

    env_maker = EnvMaker(env)
    types, args = models_config()
    args['alg'] = dict(
        env_maker=env_maker,
        n_iter=n_iter,
        n_batch=n_batch,
        n_envs=n_envs,
        gamma=gamma,
        gae_lambda=gae_lambda
    )
    config = dict(
        seed=seed,
        episodic=episodic,
        types=types,
        args=args
    )

    types['alg'] = vanilla
    train(config, proto_dir, interval=interval)

    seed = set_global_seeds(seed)
    types['alg'] = natural
    args['alg']['kl_frac'] = kl_frac
    train(config, proto_dir, interval=interval)

    seed = set_global_seeds(seed)
    types['alg'] = trpo
    args['alg']['delta'] = delta
    del types['optimizer'], args['optimizer']
    del types['scheduler'], args['scheduler']
    train(config, proto_dir, interval=interval)


if __name__ == "__main__":
    main()
