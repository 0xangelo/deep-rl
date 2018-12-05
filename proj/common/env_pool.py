import sys, gym, torch, numpy as np, multiprocessing as mp, subprocess
from .tqdm_util import tqdm, trange
from .observations import obs_to_tensor

import tblib.pickling_support
tblib.pickling_support.install()

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
    Using a pool of workers to run multiple environments in parallel. This 
    implementation supports multiple environments per worker to be as flexible 
    as possible.
    """

    def __init__(self, env_maker, n_envs=mp.cpu_count(),
                 n_parallel=mp.cpu_count()):
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
        self.last_obs = obs = np.asarray(obs, dtype=np.float32)
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
        for conn, offset, workers in zip(
                self.conns, self.worker_env_offsets, self.n_worker_envs):
            conn.send(('step', actions[offset:offset + workers]))

        results = []

        for conn in self.conns:
            status, data = conn.recv()
            if status == 'success':
                results.extend(data)
            else:
                raise data[1].with_traceback(data[2])
        next_obs, rews, dones, infos = tuple(map(list, zip(*results)))
        self.last_obs = next_obs = np.asarray(next_obs, dtype=np.float32)
        return next_obs, rews, dones, infos

    def seed(self, seeds):
        assert len(seeds) == self.n_envs
        for conn, offset, workers in zip(
                self.conns, self.worker_env_offsets, self.n_worker_envs):
            conn.send(('seed', seeds[offset:offset + workers]))
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


def parallel_collect_samples(env_pool, policy, num_samples):
    """
    Collect trajectories in parallel using a pool of workers. Actions are 
    computed using the provided policy. For each worker, \lfloor num_samples / 
    env_pool.n_workers \rfloor timesteps are sampled. This means that some of
    the trajectories will not be executed until termination. These partial 
    trajectories will have their last state index recorded in "finishes" with 
    a False flag.

    When starting, it will first check if env_pool.last_obs is set, and if so, 
    it will start from there rather than resetting all environments. This is 
    useful for reusing the same episode.

    :param env_pool: An instance of EnvPool.
    :param policy: The policy used to select actions.
    :param num_samples: The approximate total number of samples to collect.
    :return: A dictionary with all observations, actions, rewards and tuples 
    of last index, finished flag and last observation of each trajectory
    """
    offset      = num_samples // env_pool.n_envs
    num_samples = env_pool.n_envs * offset
    all_obs  = np.empty((num_samples,) + policy.ob_space.shape, dtype=np.float32)
    all_acts = np.empty((num_samples,) + policy.ac_space.shape, dtype=np.float32)
    all_rews = np.empty((num_samples,), dtype=np.float32)
    finishes = []

    obs = env_pool.reset() if env_pool.last_obs is None else env_pool.last_obs
    for idx in trange(0, offset, unit="step", leave=False, desc="Sampling"):
        actions = policy.actions(torch.as_tensor(obs)).numpy()
        next_obs, rews, dones, _ = env_pool.step(actions)
        for env in range(env_pool.n_envs):
            all_obs[env*offset + idx] = obs[env]
            all_acts[env*offset + idx] = actions[env]
            all_rews[env*offset + idx] = rews[env]
            if dones[env]:
                finishes.append(
                    (env*offset + idx + 1, True, np.zeros_like(obs[env]))
                )
        obs = next_obs
    env_pool.flush()

    for env, done in filter(lambda x: not x[1], enumerate(dones)):
        finishes.append(
            (env*offset + offset, False, obs[env])
        )

    # Ordered list with information about the ends of each trajectory
    finishes = tuple(map(list, zip(*sorted(finishes, key=lambda x: x[0]))))

    return dict(
        observations=all_obs,
        actions=all_acts,
        rewards=all_rews,
        finishes=finishes
    )


