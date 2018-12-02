import os, click, torch
from proj.algorithms import trpo, train
from proj.common.utils import set_global_seeds
from defaults import models_config


@click.command()
@click.argument("env")
@click.option("--log_dir", help="where to save checkpoint and progress data",
              type=str, default='data/')
@click.option("--episodic", help="enforce all episodes end",
              is_flag=True)
@click.option("--cuda", help="enable GPU acceleration if available",
              is_flag=True)
@click.option("--interval", help="interval between each snapshot",
              type=int, default=10)
@click.option("--seed", help="for repeatability",
              type=int, default=None)
@click.option("--model", help="which model configuration to use",
              type=str, default='Mlp:64-64:elu')
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
@click.option("--kl_frac", help="fraction of samples for kl computation",
              type=float, default=0.4)
@click.option("--delta", help="kl divergence constraint per step",
              type=float, default=0.01)
def main(env, episodic, cuda, log_dir, interval, seed, model, **algargs):
    """Runs TRPO on given environment with specified parameters."""
    
    seed = set_global_seeds(seed)
    if cuda and torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.FloatTensor)
    log_dir = os.path.join(log_dir, env, model + '+None', 'trpo', str(seed), '')

    params = dict(
        exp_name='trpo',
        env=env,
        episodic=episodic,
        seed=seed,
        model=model,
        **algargs
    )
    types, args = models_config(model)
    types['alg'], args['alg'] = trpo, algargs

    train(params, types, args, log_dir=log_dir, interval=interval)

if __name__ == "__main__":
    main()
