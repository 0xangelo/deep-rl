import click
import time
from proj.common.utils import SnapshotSaver


@click.command()
@click.argument("dir")
@click.option("--index", help="Wich file to load from", type=int, default=None)
def main(dir, index):
    state = None
    while state is None:
        saver = SnapshotSaver(dir)
        state = saver.get_state(index)
        if state is None:
            time.sleep(1)
        
    alg_state = state['alg_state']
    env = alg_state['env_maker'].make()
    policy = alg_state['policy']
    for _ in range(10):
        rewards = 0
        ob = env.reset()
        done = False
        while not done:
            action, _ = policy.get_action(ob)
            ob, rew, done, _ = env.step(action)
            env.render()
            rewards += rew
        print(rewards)
    env.close()


if __name__ == "__main__":
    main()
