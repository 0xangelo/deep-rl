from contextlib import suppress

import gym
import proj  # pylint: disable=unused-import


def main():
    import argparse

    parse = argparse.ArgumentParser()
    parse.add_argument("--env", "-e", default="CartPoleSwingUp-v0")
    args = parse.parse_args()
    with suppress(KeyboardInterrupt), gym.make(args.env) as world:
        simulate(world)


def simulate(env):
    env.reset()
    done = False
    while True:
        _, _, done, _ = env.step(env.action_space.sample())
        env.render()
        if done:
            env.reset()
            done = False


if __name__ == "__main__":
    main()
