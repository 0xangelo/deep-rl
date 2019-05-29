import click
import time
import torch
import pprint
from baselines import logger
from proj.utils.saver import SnapshotSaver
from proj.common.log_utils import log_reward_statistics
from proj.common.env_makers import VecEnvMaker


@click.command()
@click.argument("path")
@click.option("--index", help="Wich checkpoint to load from",
              type=int, default=None)
@click.option("--runs", help="Number of episodes to simulate",
              type=int, default=2)
@click.option("--norender", help="Don't render the simulation on screen",
              is_flag=True)
@click.option("--deterministic", help="Use mode of the distributions if applicable",
              is_flag=True)
@click.option("--env", help="Override which environment the policy will be executed in",
              type=str, default=None)
def main(path, index, runs, norender, deterministic, env):
    """
    Loads a snapshot and simulates the corresponding policy and environment.
    """

    snapshot = None
    saver = SnapshotSaver(path)
    while snapshot is None:
        snapshot = saver.get_state(index)
        if snapshot is None:
            time.sleep(1)

    config, state = snapshot
    pprint.pprint(config)
    if env is not None:
        env = VecEnvMaker(env)(train=False)
    else:
        env = VecEnvMaker(config['env'])(train=False)
    policy = config['policy'].pop('class')(env, **config['policy'])
    policy.load_state_dict(state['policy'])

    with torch.no_grad():
        if deterministic:
            policy.eval()
        for _ in range(runs):
            ob = env.reset()
            done = False
            while not done:
                action = policy.actions(torch.from_numpy(ob))
                ob, _, done, _ = env.step(action.numpy())
                if not norender:
                    env.render()

    env.close()
    log_reward_statistics(env)
    logger.dumpkvs()

if __name__ == "__main__":
    main()
