import os
import sys
import gym
import multiprocessing as mp
import subprocess
import numpy as np
import torch
import time
import random
import cloudpickle
import tblib.pickling_support

from tqdm import trange
from proj.common import logger
from proj.common.tqdm_out import term

tblib.pickling_support.install()


def set_global_seeds(i):
    np.random.seed(i)
    random.seed(i)
    torch.manual_seed(i)


def flatten_dim(space):
    if isinstance(space, gym.spaces.Box):
        return np.prod(space.shape)
    elif isinstance(space, gym.spaces.Discrete):
        return space.n
    else:
        assert False


def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    """
    Demmel p 312. Approximately solve x = A^{-1}b, or Ax = b, where we only have access to f: x -> Ax
    """
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)

    for i in range(cg_iters):
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    return x


def explained_variance_1d(ypred, y):
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    if np.isclose(vary, 0):
        if np.var(ypred) > 1e-8:
            return 0
        else:
            return 1
    return 1 - np.var(y - ypred) / (vary + 1e-8)


# ==============================
# Parallel traj collecting
# ==============================

def env_worker(env_maker, conn, n_worker_envs):
    envs = []
    for _ in range(n_worker_envs):
        envs.append(env_maker.make())
    while True:
        command, data = conn.recv()
        try:
            if command == 'reset':
                obs = []
                for env in envs:
                    obs.append(env.reset())
                conn.send(('success', obs))
            elif command == 'seed':
                seeds = data
                for env, seed in zip(envs, seeds):
                    env.seed(seed)
                conn.send(('success', None))
            elif command == 'step':
                actions = data
                results = []
                for env, action in zip(envs, actions):
                    next_ob, rew, done, info = env.step(action)
                    if done:
                        info["last_observation"] = next_ob
                        next_ob = env.reset()
                    results.append((next_ob, rew, done, info))
                conn.send(('success', results))
            elif command == 'flush':
                for env in envs:
                    env._flush(True)
                conn.send(('success', None))
            elif command == 'close':
                for env in envs:
                    env.close()
                conn.send(('success', None))
                return
            else:
                raise ValueError("Unrecognized command: {}".format(command))
        except Exception as e:
            conn.send(('error', sys.exc_info()))


class EnvPool(object):
    """
    Using a pool of workers to run multiple environments in parallel. This implementation supports multiple environments
    per worker to be as flexible as possible.
    """

    def __init__(self, env_maker, n_envs=mp.cpu_count(), n_parallel=mp.cpu_count()):
        self.env_maker = env_maker
        self.n_envs = n_envs
        # No point in having more parallel workers than environments
        if n_parallel > n_envs:
            n_parallel = n_envs
        self.n_parallel = n_parallel
        self.workers = []
        self.conns = []
        # try to split evenly, but this isn't always possible
        self.n_worker_envs = [len(d) for d in np.array_split(
            np.arange(self.n_envs), self.n_parallel)]
        self.worker_env_offsets = np.concatenate(
            [[0], np.cumsum(self.n_worker_envs)[:-1]])
        self.last_obs = None

    def start(self):
        workers = []
        conns = []
        for idx in range(self.n_parallel):
            worker_conn, master_conn = mp.Pipe()
            worker = mp.Process(target=env_worker, args=(
                self.env_maker, worker_conn, self.n_worker_envs[idx]))
            worker.start()
            # pin each worker to a single core
            if sys.platform == 'linux':
                subprocess.check_call(
                    ["taskset", "-p", "-c",
                        str(idx % mp.cpu_count()), str(worker.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            workers.append(worker)
            conns.append(master_conn)

        self.workers = workers
        self.conns = conns

        # set initial seeds
        seeds = np.random.randint(
            low=0, high=np.iinfo(np.int32).max, size=self.n_envs)
        self.seed([int(x) for x in seeds])

    def __enter__(self):
        self.start()
        return self

    def reset(self):
        for conn in self.conns:
            conn.send(('reset', None))
        obs = []
        for conn in self.conns:
            status, data = conn.recv()
            if status == 'success':
                obs.extend(data)
            else:
                raise data[1].with_traceback(data[2])
        assert len(obs) == self.n_envs
        self.last_obs = obs
        return obs

    def flush(self):
        for conn in self.conns:
            conn.send(('flush', None))
        for conn in self.conns:
            status, data = conn.recv()
            if status != 'success':
                raise data[1].with_traceback(data[2])

    def step(self, actions):
        assert len(actions) == self.n_envs
        for idx, conn in enumerate(self.conns):
            offset = self.worker_env_offsets[idx]
            conn.send(
                ('step', actions[offset:offset + self.n_worker_envs[idx]]))

        results = []

        for conn in self.conns:
            status, data = conn.recv()
            if status == 'success':
                results.extend(data)
            else:
                raise data[1].with_traceback(data[2])
        next_obs, rews, dones, infos = list(map(list, zip(*results)))
        self.last_obs = next_obs
        return next_obs, rews, dones, infos

    def seed(self, seeds):
        assert len(seeds) == self.n_envs
        for idx, conn in enumerate(self.conns):
            offset = self.worker_env_offsets[idx]
            conn.send(('seed', seeds[offset:offset + self.n_worker_envs[idx]]))
        for conn in self.conns:
            status, data = conn.recv()
            if status != 'success':
                raise data[1].with_traceback(data[2])

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        for conn in self.conns:
            conn.send(('close', None))
        for conn in self.conns:
            status, data = conn.recv()
            if status != 'success':
                raise data[1].with_traceback(data[2])
        for worker in self.workers:
            worker.join()
        self.workers = []
        self.conns = []


def _cast_and_add(trajs, traj):
    trajs.append(
        dict(
            observations=np.asarray(traj["observations"]),
            actions=np.asarray(traj["actions"]),
            rewards=np.asarray(traj["rewards"], dtype=np.float32),
            distributions=np.asarray(traj["distributions"]),
            last_observation=traj["last_observation"],
            finished=traj["finished"],
        )
    )
    
def parallel_collect_samples(env_pool, policy, num_samples):
    """
    Collect trajectories in parallel using a pool of workers. Actions are computed using the provided policy.
    Collection will continue until at least num_samples trajectories are collected. It will exceed this amount by
    at most env_pool.n_envs. This means that some of the trajectories will not be executed until termination. These
    partial trajectories will have their "finished" entry set to False.

    When starting, it will first check if env_pool.last_obs is set, and if so, it will start from there rather than
    resetting all environments. This is useful for reusing the same episode.

    :param env_pool: An instance of EnvPool.
    :param policy: The policy used to select actions.
    :param num_samples: The minimum number of samples to collect.
    :return: A list of trajectories, each a dictionary.
    """
    trajs = []
    partial_trajs = [None] * env_pool.n_envs

    if env_pool.last_obs is not None:
        obs = env_pool.last_obs
    else:
        obs = env_pool.reset()

    for _ in trange(0, num_samples, env_pool.n_envs, desc="Sampling",
                    unit="steps", leave=False, file=term(), dynamic_ncols=True):
        actions, dists = policy.get_actions(obs)
        next_obs, rews, dones, infos = env_pool.step(actions)
        for idx in range(env_pool.n_envs):
            if partial_trajs[idx] is None:
                partial_trajs[idx] = dict(
                    observations=[],
                    actions=[],
                    rewards=[],
                    distributions=[],
                )
            traj = partial_trajs[idx]
            traj["observations"].append(obs[idx])
            traj["actions"].append(actions[idx])
            traj["rewards"].append(rews[idx])
            traj["distributions"].append(dists[idx])
            if dones[idx]:
                traj["last_observation"] = infos[idx]["last_observation"]
                traj["finished"] = True
                _cast_and_add(trajs, traj)
                partial_trajs[idx] = None
        obs = next_obs

    for idx in range(env_pool.n_envs):
        if partial_trajs[idx] is not None:
            traj = partial_trajs[idx]
            traj["last_observation"] = obs[idx]
            traj["finished"] = False
            _cast_and_add(trajs, traj)
    env_pool.flush()
    return trajs


# ==============================
# Saving snapshots
# ==============================

class SnapshotSaver(object):
    def __init__(self, dir, interval=1, latest_only=None):
        self.dir = dir
        self.interval = interval
        if latest_only is None:
            latest_only = True
            snapshots_folder = os.path.join(dir, "snapshots")
            if os.path.exists(snapshots_folder):
                if os.path.exists(os.path.join(snapshots_folder, "latest.pkl")):
                    latest_only = True
                elif len(os.listdir(snapshots_folder)) > 0:
                    latest_only = False
        self.latest_only = latest_only

    @property
    def snapshots_folder(self):
        return os.path.join(self.dir, "snapshots")

    def get_snapshot_path(self, index):
        if self.latest_only:
            return os.path.join(self.snapshots_folder, "latest.pkl")
        else:
            return os.path.join(self.snapshots_folder, "%d.pkl" % index)

    def save_state(self, index, state):
        if index % self.interval == 0:
            file_path = self.get_snapshot_path(index)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as f:
                cloudpickle.dump(state, f, protocol=-1)

    def get_state(self, index=None):
        if self.latest_only:
            try:
                with open(self.get_snapshot_path(0), "rb") as f:
                    return cloudpickle.load(f)
            except EOFError:
                pass
        else:
            if index is not None:
                try:
                    with open(self.get_snapshot_path(index), "rb") as f:
                        return cloudpickle.load(f)
                except EOFError:
                    pass
            snapshot_files = os.listdir(self.snapshots_folder)
            snapshot_files = sorted(
                snapshot_files, key=lambda x: int(x.split(".")[0]))[::-1]
            for file in snapshot_files:
                file_path = os.path.join(self.snapshots_folder, file)
                try:
                    with open(file_path, "rb") as f:
                        return cloudpickle.load(f)
                except EOFError:
                    pass

