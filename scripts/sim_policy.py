import torch
import click
import time
from proj.common.saver import SnapshotSaver


@click.command()
@click.argument("path")
@click.option("--index", help="Wich checkpoint to load from", type=int, default=None)
def main(path, index):
    """
    Loads a snapshot and simulates the corresponding policy and environment.
    """
    
    state = None
    while state is None:
        saver = SnapshotSaver(path)
        state = saver.get_state(index)
        if state is None:
            time.sleep(1)
        
    alg_state = state['alg_state']
    env = alg_state['env_maker'].make(pytorch=True)
    policy = alg_state['policy']
    for _ in range(10):
        rewards = 0
        ob = env.reset()
        done = False
        while not done:
            action = policy.action(ob)
            ob, rew, done, _ = env.step(action)
            env.render()
            rewards += rew
        print(rewards)
    env.unwrapped.close()


if __name__ == "__main__":
    main()
