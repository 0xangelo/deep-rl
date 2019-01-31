import os
import gym
from baselines import logger
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from gym.envs.atari.atari_env import AtariEnv


class EnvMaker(object):
    def __init__(self, env_id):
        self.env_id = env_id
        self.__name__ = env_id

    def __call__(self):
        if 'AtariEnv' in gym.spec(self.env_id)._entry_point \
           and '-ram-' not in self.env_id:
            env = make_atari(self.env_id)
        else:
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

        if isinstance(env.unwrapped, AtariEnv) and '-ram-' not in self.env_id:
            env = wrap_deepmind(env, frame_stack=True)
        return env
