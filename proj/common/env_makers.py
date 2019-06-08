"""
Implements several factories for both single and vectorized environments.
"""
import os
import gym
import numpy as np
from baselines import logger
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv as _DummyVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.vec_env.vec_monitor import VecMonitor
from proj.common.env_pool import EnvPool, ShmEnvPool


class EnvMaker:
    """
    Used to store an environment id and apply the appropriate wrappers
    when constructing the environment.
    """

    def __init__(self, env_id):
        self.env_id = env_id
        self.__name__ = repr(self)

    def __call__(self):
        if (
            "AtariEnv" in gym.spec(self.env_id)._entry_point
            and "-ram-" not in self.env_id
        ):
            env = make_atari(self.env_id)
            env = wrap_deepmind(env)
        else:
            env = gym.make(self.env_id)

        if len(env.observation_space.shape) == 1 and "TimeLimit" in str(env):
            env = AddRelativeTimestep(env)

        return env

    def __repr__(self):
        return "EnvMaker('{}')".format(self.env_id)


class VecEnvMaker:
    """
    Used to store an environment id and apply the appropriate vectorized
    environment wrappers when constructing the vectorized environment.
    """

    def __init__(self, env_id):
        self.env_id = env_id
        self.__name__ = repr(self)

    def __call__(self, n_envs=1, *, train=True):
        env_fn = EnvMaker(self.env_id)

        if (
            "AtariEnv" in gym.spec(self.env_id)._entry_point
            and "-ram-" not in self.env_id
        ):
            if n_envs == 1:
                vec_env = DummyVecEnv([env_fn])
            else:
                vec_env = ShmEnvPool(env_fn, n_envs=n_envs)
            vec_env = VecFrameStack(vec_env, 4)
        else:
            if n_envs == 1:
                vec_env = DummyVecEnv([env_fn])
            else:
                vec_env = EnvPool(env_fn, n_envs=n_envs)

        monitor_dir = os.path.join(
            logger.get_dir(), ("train" if train else "eval") + "_monitor"
        )
        os.makedirs(monitor_dir, exist_ok=True)
        vec_env = VecMonitor(vec_env, filename=monitor_dir)
        return vec_env

    def __repr__(self):
        return "VecEnvMaker('{}')".format(self.env_id)


# ==============================
# Reproducible DummyVecEnv
# ==============================


class DummyVecEnv(_DummyVecEnv):
    """
    Extends baselines.common.vec_env.dummy_vec_env.DummyVecEnv to allow seeding
    and perform it on initialization.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # set initial seeds
        seeds = np.random.randint(
            low=0, high=np.iinfo(np.int32).max, size=self.num_envs
        )
        self.seed(seeds)

    def seed(self, seeds):
        for env, seed in zip(self.envs, seeds):
            env.seed(int(seed))


# ==============================
# Wrappers
# ==============================


class AddTimestep(gym.ObservationWrapper):
    """
    Adds the absolute timestep to the observations of environments with
    observation spaces of type gym.spaces.Box.
    """

    def __init__(self, env=None):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=np.append(self.observation_space.low, 0),
            high=np.append(self.observation_space.high, 2 ** 32),
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation):
        return np.append(observation, self.env._elapsed_steps)


class AddRelativeTimestep(gym.ObservationWrapper):
    """
    Adds the relative timestep (normalized to the range [-1, 1]) to the
    observations of environments with observation spaces of type gym.spaces.Box.
    """

    def __init__(self, env=None):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=np.append(self.observation_space.low, -1.0),
            high=np.append(self.observation_space.high, 1.0),
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation):
        return np.append(
            observation, -1 + (self.env._elapsed_steps / self.spec.timestep_limit) * 2
        )
