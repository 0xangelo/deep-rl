import os
import gym
from baselines import logger
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.vec_env.vec_monitor import VecMonitor
from proj.common.env_pool import EnvPool, ShmEnvPool


class EnvMaker(object):
    def __init__(self, env_id):
        self.env_id = env_id
        self.__name__ = env_id

    def __call__(self):
        if 'AtariEnv' in gym.spec(self.env_id)._entry_point \
           and '-ram-' not in self.env_id:
            env = make_atari(self.env_id)
            env = wrap_deepmind(env)
        else:
            env = gym.make(self.env_id)

        return env


class VecEnvMaker(object):
    def __init__(self, env_id):
        self.env_id = env_id
        self.__name__ = env_id

    def __call__(self, n_envs=1):
        env_fn = EnvMaker(self.env_id)

        if n_envs == 1:
            vec_env = DummyVecEnv([env_fn])
        elif 'AtariEnv' in gym.spec(self.env_id)._entry_point \
           and '-ram-' not in self.env_id:
            vec_env = ShmEnvPool(env_fn, n_envs=n_envs)
            vec_env = VecFrameStack(vec_env, 4)
        else:
            vec_env = VecEnvPool(env_fn, n_envs=n_envs)

        monitor_dir = os.path.join(logger.get_dir(), "bench_monitor", '')
        os.makedirs(monitor_dir, exist_ok=True)
        vec_env = VecMonitor(vec_env, filename=monitor_dir)
        setattr(vec_env, 'directory', os.path.abspath(monitor_dir))
        return vec_env
