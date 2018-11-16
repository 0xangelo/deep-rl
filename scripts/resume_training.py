import os, torch, click, time, json
from proj.common.utils import set_global_seeds
from proj.common import logger
from proj.common.tqdm_util import tqdm_out
from proj.common.env_makers import EnvMaker
from proj.common.saver import SnapshotSaver
from proj.common import env_pool


@click.command()
@click.argument("path")
@click.option("--index", help="Wich checkpoint to load from",
              type=int, default=None)
@click.option("--add", help="Number of additional iterations to run for",
              type=int, default=0)
def main(path, index, add):
    """
    Loads latest snapshot and resumes training.

    WARNING: not repeatable, rng state cannot be recovered from snapshot
    """

    with open(os.path.join(path, 'variant.json')) as f:
        params = json.load(f)
    env_pool.episodic = True if params['episodic'] else False

    state = None
    while state is None:
        saver = SnapshotSaver(path)
        config, state = saver.get_state(index)
        if state is None:
            time.sleep(1)
    del saver
    
    types, args = config

    alg, kwargs = types['alg'], args['alg']
    kwargs.update(state['alg'])
    kwargs['n_iter'] += add
    kwargs['env_maker'] = env_maker = EnvMaker(params['env'])
    saver = SnapshotSaver(path, (types, args), interval=10)
    with tqdm_out(), logger.session(path):
        env = env_maker.make()

        policy = types['policy'](env, **args['policy'])
        policy.load_state_dict(state['policy'])
        baseline = types['baseline'](env, **args['baseline'])
        baseline.load_state_dict(state['baseline'])

        if 'optimizer' in types:
            optimizer = types['optimizer'](
                policy.parameters(), **args['optimizer']
            )
            optimizer.load_state_dict(state['optimizer'])
            scheduler = types['scheduler'](optimizer, **args['scheduler'])
            scheduler.load_state_dict(state['scheduler'])

            alg(
                env=env,
                policy=policy,
                baseline=baseline,
                optimizer=optimizer,
                scheduler=scheduler,
                snapshot_saver=saver,
                **kwargs
            )
        else:
            alg(
                env=env,
                policy=policy,
                baseline=baseline,
                snapshot_saver=saver,
                **kwargs
            )
            

if __name__ == "__main__":
    main()
