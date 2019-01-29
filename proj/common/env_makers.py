import os
import gym
from baselines import logger
from baselines.common.atari_wrappers import wrap_deepmind
from gym.envs.atari.atari_env import AtariEnv


class EnvMaker(object):
    def __init__(self, env_id):
        self.env_id = env_id
        self.__name__ = env_id

    def __call__(self):
        env = gym.make(self.env_id)
        monitor_dir = os.path.join(logger.get_dir(), "gym_monitor")
        if logger.Logger.CURRENT is not logger.Logger.DEFAULT:
            resume = True
            force = False
        else:
            resume = False
            force = True
        env = gym.wrappers.Monitor(env, directory=monitor_dir, force=force,
                                   resume=resume, video_callable=False)
        if isinstance(env.unwrapped, AtariEnv):
            if '-ram-' in self.env_id:
                assert 'NoFrameskip' not in self.env_id
            else:
                env = wrap_deepmind(env)
        return env
