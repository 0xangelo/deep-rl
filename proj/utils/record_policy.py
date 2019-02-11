import click
import time
import torch
import pprint
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from proj.utils.saver import SnapshotSaver


@click.command()
@click.argument("exp_path")
@click.argument("vid_path")
@click.option("--n_envs", help="Number of environments to run in parallel",
              type=int, default=1)
@click.option("--steps", help="Number of steps to record",
              type=int, default=1000)
def main(exp_path, vid_path, n_envs, steps):
    """
    Loads a snapshot and simulates the corresponding policy and environment.
    """

    if ':' in exp_path:
        exp_path, index = exp_path.split(':')
    else:
        index = None

    snapshot = None
    saver = SnapshotSaver(exp_path)
    while snapshot is None:
        snapshot = saver.get_state(index)
        if snapshot is None:
            time.sleep(1)

    config, state = snapshot
    pprint.pprint(config)
    vec_env = config['env_maker'](n_envs)
    vec_env = VecVideoRecorder(
        vec_env, vid_path, lambda _: True, video_length=steps)
    policy = config['policy'].pop('class')(vec_env, **config['policy'])
    policy.load_state_dict(state['policy'])

    ob = vec_env.reset()
    for _ in range(steps):
        action = policy.actions(torch.from_numpy(ob))
        ob, _, done, _ = vec_env.step(action.numpy())

    vec_env.close()

if __name__ == "__main__":
    main()
