import time
import pprint
from contextlib import suppress

import click
import torch
from baselines import logger
from proj.utils.saver import SnapshotSaver
from proj.common.log_utils import log_reward_statistics
from proj.common.env_makers import VecEnvMaker


@click.command()
@click.argument("path")
@click.option("--index", type=int, default=None, help="Wich checkpoint to load from")
@click.option(
    "--render/--no-render",
    default=True,
    help="Whether or not to render the simulation on screen",
)
@click.option(
    "--deterministic",
    "-d",
    is_flag=True,
    help="Use mode of the distributions if applicable",
)
@click.option(
    "--env",
    "-e",
    default=None,
    help="Override which environment the policy will be executed in",
)
def main(**args):  # path, index, runs, norender, deterministic, env):
    """
    Loads a snapshot and simulates the corresponding policy and environment.
    """
    snapshot = None
    saver = SnapshotSaver(args["path"])
    while snapshot is None:
        snapshot = saver.get_state(args["index"])
        if snapshot is None:
            time.sleep(1)

    config, state = snapshot
    pprint.pprint(config)
    if args["env"] is not None:
        env = VecEnvMaker(args["env"])(train=False)
    else:
        env = VecEnvMaker(config["env"])(train=False)
    policy = config["policy"].pop("class")(env, **config["policy"])
    policy.load_state_dict(state["policy"])
    if args["deterministic"]:
        policy.eval()

    with torch.no_grad(), suppress(KeyboardInterrupt):
        simulate(env, policy, render=args["render"])

    env.close()
    log_reward_statistics(env)
    logger.dumpkvs()


def simulate(env, policy, render=True):
    obs = env.reset()
    while True:
        action = policy.actions(torch.from_numpy(obs))
        obs, _, _, _ = env.step(action.numpy())
        if render:
            env.render()


if __name__ == "__main__":
    main()
