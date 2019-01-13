import gym, click, time, numpy as np, pprint
from proj.utils.saver import SnapshotSaver


@click.command()
@click.argument("path")
@click.option("--index", help="Wich checkpoint to load from",
              type=int, default=None)
@click.option("--runs", help="Number of episodes to simulate",
              type=int, default=2)
@click.option("--norender", help="""Whether or not to render the
              simulation on screen""", is_flag=True)
def main(path, index, runs, norender):
    """
    Loads a snapshot and simulates the corresponding policy and environment.
    """

    state = None
    while state is None:
        saver = SnapshotSaver(path, latest_only=(index is None))
        config, state = saver.get_config(), saver.get_state(index)
        if config is None or state is None:
            time.sleep(1)

    pprint.pprint(config)
    env = config['env_maker'].make(pytorch=True)
    policy = config['policy'].pop('class')(env, **config['policy'])
    policy.load_state_dict(state['policy'])

    for _ in range(runs):
        ob = env.reset()
        done = False
        while not done:
            action = policy.action(ob)
            ob, rew, done, _ = env.step(action)
            if not norender: env.render()

    # keep unwrapping until we get the monitor
    while not isinstance(env, gym.wrappers.Monitor):
        if not isinstance(env, gym.Wrapper):
            assert False
        env = env.env
    assert isinstance(env, gym.wrappers.Monitor)
    if runs > 10:
        print("Average total reward:", np.mean(env.get_episode_rewards()))
        print("Average episode length:", np.mean(env.get_episode_lengths()))
    else:
        print("Episode Rewards:", env.get_episode_rewards())
        print("Episode Lengths:", env.get_episode_lengths())
    env.unwrapped.close()


if __name__ == "__main__":
    main()
