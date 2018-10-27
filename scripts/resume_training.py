import torch
import click
import time
from proj.common import logger
from proj.common.saver import SnapshotSaver
from proj.common.models import restore_models
from proj.common.tqdm_util import tqdm_out
import proj.common.env_pool as pool


@click.command()
@click.argument("path")
@click.option("--add", help="Number of additional iterations to run for", type=int, default=0)
def main(path, add):
    """
    Loads latest snapshot and resumes training.

    WARNING: not repeatable, rng state cannot be recovered from snapshot
    """

    state = None
    while state is None:
        saver = SnapshotSaver(path)
        state = saver.get_state()
        if state is None:
            time.sleep(1)

    with tqdm_out(), logger.session(path):
        alg = state['alg']
        alg_state = state['alg_state']
        
        env = alg_state['env_maker'].make()
        alg_state['n_iter'] += add
        alg_state.update(restore_models(state['models'], env))

        alg(
            env=env,
            snapshot_saver=saver,
            **alg_state
        )

if __name__ == "__main__":
    main()
