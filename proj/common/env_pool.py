import sys, gym, torch, numpy as np, multiprocessing as mp, subprocess
from .tqdm_util import trange
from .observations import obs_to_tensor

import tblib.pickling_support
tblib.pickling_support.install()


BOOTSTRAP = True

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
                dones = [False] * n_worker_envs
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
            elif command == 'finish':
                to_run = filter(lambda x: not x[2], zip(envs, data[0], data[1]))
                results = []
                for env, action, _ in to_run:
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
    Using a pool of workers to run multiple environments in parallel. This 
    implementation supports multiple environments per worker to be as flexible 
    as possible.
    """

    def __init__(self, env, env_maker, n_envs=mp.cpu_count(),
                 n_parallel=mp.cpu_count()):
        self.env_maker = env_maker
        self.obs_to_tensor = obs_to_tensor(env.observation_space)
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
        self.dones = None

    def start(self):
        workers = []
        conns = []
        for idx in range(self.n_parallel):
            worker_conn, master_conn = mp.Pipe()
            worker = mp.Process(target=env_worker, args=(
                self.env_maker, worker_conn, self.n_worker_envs[idx]))
            worker.start()
            # # pin each worker to a single core
            # if sys.platform == 'linux':
            #     subprocess.check_call(
            #         ["taskset", "-p", "-c",
            #             str(idx % mp.cpu_count()), str(worker.pid)],
            #         stdout=subprocess.DEVNULL,
            #         stderr=subprocess.DEVNULL,
            #     )
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
        obs = self.obs_to_tensor(obs)
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
        actions = actions.cpu().numpy()
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
        next_obs, rews, dones, infos = tuple(map(list, zip(*results)))
        next_obs = self.obs_to_tensor(next_obs)
        self.last_obs, self.dones = next_obs, torch.tensor(dones)
        return next_obs, rews, dones, infos

    def finish(self, actions):
        empty = torch.empty(
            (self.n_envs,) + actions[0].shape, dtype=actions.dtype)
        empty[self.dones == 0] = actions
        empty, dones = empty.cpu().numpy(), self.dones.cpu().numpy()
        for idx, conn in enumerate(self.conns):
            offset = self.worker_env_offsets[idx]
            conn.send((
                'finish',
                (empty[offset:offset + self.n_worker_envs[idx]],
                dones[offset:offset + self.n_worker_envs[idx]])
            ))

        results = []

        for conn in self.conns:
            status, data = conn.recv()
            if status == 'success':
                results.extend(data)
            else:
                raise data[1].with_traceback(data[2])
        next_obs, rews, dones, infos = tuple(map(list, zip(*results)))
        self.last_obs[self.dones == 0] = self.obs_to_tensor(next_obs)
        self.dones[self.dones == 0] = torch.tensor(dones)
        return self.last_obs[self.dones == 0], rews, dones, infos

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


def parallel_collect_samples(env_pool, policy, num_samples):
    """
    Collect trajectories in parallel using a pool of workers. Actions are 
    computed using the provided policy. Collection will continue until at least 
    num_samples trajectories are collected. It will exceed this amount by at 
    most env_pool.n_envs. This means that some of the trajectories will not be 
    executed until termination. These partial trajectories will have their 
    "finished" entry set to False.

    When starting, it will first check if env_pool.last_obs is set, and if so, 
    it will start from there rather than resetting all environments. This is 
    useful for reusing the same episode.

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

    for _ in trange(0, num_samples, env_pool.n_envs, unit="step", leave=False,
                    desc="Sampling", dynamic_ncols=True):
        actions = policy.actions(obs)
        next_obs, rews, dones, infos = env_pool.step(actions)
        for idx, traj in enumerate(partial_trajs):
            if traj is None:
                partial_trajs[idx] = traj = dict(
                    observations=[],
                    actions=[],
                    rewards=[],
                )
            traj["observations"].append(obs[idx])
            traj["actions"].append(actions[idx])
            traj["rewards"].append(rews[idx])
            if dones[idx]:
                trajs.append(
                    dict(
                        observations=torch.stack(traj["observations"]),
                        actions=torch.stack(traj["actions"]),
                        rewards=torch.Tensor(traj["rewards"]),
                        last_observation=env_pool.obs_to_tensor(
                            infos[idx]["last_observation"]),
                        finished=True,
                    )
                )
                partial_trajs[idx] = None
        obs = next_obs

    if BOOTSTRAP:
        for idx, traj in enumerate(partial_trajs):
            if traj is None: continue
            trajs.append(
                dict(
                    observations=torch.stack(traj["observations"]),
                    actions=torch.stack(traj["actions"]),
                    rewards=torch.Tensor(traj["rewards"]),
                    last_observation=obs[idx],
                    finished=False,
                )
            )

    else:
        obs = env_pool.last_obs[env_pool.dones == 0]
        while len(obs) > 0:
            actions = policy.actions(obs)
            next_obs, rews, dones, infos = env_pool.finish(actions)
            num = 0
            for idx, traj in enumerate(partial_trajs):
                if traj is None: continue
                traj["observations"].append(obs[num])
                traj["actions"].append(actions[num])
                traj["rewards"].append(rews[num])
                if dones[num]:
                    trajs.append(
                        dict(
                            observations=torch.stack(traj["observations"]),
                            actions=torch.stack(traj["actions"]),
                            rewards=torch.Tensor(traj["rewards"]),
                            last_observation=env_pool.obs_to_tensor(
                                infos[num]["last_observation"]),
                            finished=True,
                        )
                    )
                    partial_trajs[idx] = None
                num += 1
            obs = next_obs


    env_pool.flush()
    return trajs


