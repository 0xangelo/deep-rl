import os
from ..common import logger
from ..common.tqdm_util import tqdm_out
from ..common.saver import SnapshotSaver
from ..common import env_pool

def train(config, proto_dir, interval=1):
    types, args = config['types'], config['args']
    alg, kwargs = types['alg'], args['alg']
    
    env_pool.episodic = True if config['episodic'] else False
    seed = config['seed']
    log_dir = proto_dir.format(alg=alg.__name__, seed=seed)
    variant = dict(exp_name=alg.__name__, seed=seed)
    os.system("rm -rf {}".format(log_dir))
    saver = SnapshotSaver(log_dir, config, interval=interval)

    with tqdm_out(), logger.session(log_dir, variant=variant):
        env = kwargs['env_maker'].make()
        policy = types['policy'](env, **args['policy'])
        baseline = types['baseline'](env, **args['baseline'])
        if 'optimizer' in types:
            optimizer = types['optimizer'](
                policy.parameters(), **args['optimizer'])
            scheduler = types['scheduler'](
                optimizer, **args['scheduler'])

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
            
