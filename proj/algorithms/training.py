import os
from ..common import logger
from ..common.tqdm_util import tqdm_out
from ..common.env_makers import EnvMaker
from ..common.saver import SnapshotSaver
from ..common import env_pool

def train(params, types, args, log_dir='/tmp/experiment/', interval=1):
    os.system("rm -rf {}".format(log_dir))

    alg, kwargs = types['alg'], args['alg']
    kwargs['env_maker'] = env_maker = EnvMaker(params['env'])
    saver = SnapshotSaver(log_dir, (types, args), interval=interval)
    with tqdm_out(), logger.session(log_dir, variant=params):
        env = env_maker.make()
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
            
