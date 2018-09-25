import gym, torch, click, time
from proj.common.saver import SnapshotSaver


@click.command()
@click.argument("path")
@click.option("--index", help="Wich checkpoint to load from", type=int, default=None)
@click.option("--runs", help="Number of episodes to simulate", type=int, default=2)
def main(path, index, runs):
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
    for _ in range(runs):
        ob = env.reset()
        done = False
        while not done:
            action = policy.action(ob)
            ob, rew, done, _ = env.step(action)
            env.render()

    # keep unwrapping until we get the monitor
    while not isinstance(env, gym.wrappers.Monitor):
        if not isinstance(env, gym.Wrapper):
            assert False
        env = env.env
    assert isinstance(env, gym.wrappers.Monitor)
    print("Episode Rewards:", env.get_episode_rewards())
    print("Episode Lengths:", env.get_episode_lengths())
    env.unwrapped.close()


if __name__ == "__main__":
    main()
