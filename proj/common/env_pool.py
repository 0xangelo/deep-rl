import numpy as np, multiprocessing as mp


# ==============================
# Parallel traj collecting
# ==============================

def env_worker(env_maker, conn, n_envs):
    envs = [env_maker.make() for _ in range(n_envs)]
    try:
        while True:
            command, data = conn.recv()
            if command == 'reset':
                conn.send([env.reset() for env in envs])
            elif command == 'seed':
                for env, seed in zip(envs, data): env.seed(seed)
            elif command == 'step':
                results = []
                for env, action in zip(envs, data):
                    next_ob, rew, done, info = env.step(action)
                    if done: next_ob = env.reset()
                    results.append((next_ob, rew, done, info))
                conn.send(results)
            elif command == 'flush':
                for env in envs: env._flush(True)
            elif command == 'close':
                conn.close()
                break
            else:
                raise ValueError("Unrecognized command: {}".format(command))
    except KeyboardInterrupt:
        print('EnvPool worker: ')
    finally:
        for env in envs: env.close()


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
        self.n_worker_envs = [
            len(d) for d in np.array_split(np.arange(n_envs), n_parallel)]
        self.worker_env_offsets = np.concatenate(
            [[0], np.cumsum(self.n_worker_envs)[:-1]])
        self.last_obs = None

    def start(self):
        workers = []
        conns = []
        for n_envs in self.n_worker_envs:
            worker_conn, master_conn = mp.Pipe()
            worker = mp.Process(
                target=env_worker, args=(self.env_maker, worker_conn, n_envs))
            worker.daemon = True
            worker.start()
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
            obs.extend(conn.recv())
        assert len(obs) == self.n_envs
        self.last_obs = obs = np.asarray(obs, dtype=np.float32)
        return obs

    def flush(self):
        for conn in self.conns:
            conn.send(('flush', None))

    def step(self, actions):
        assert len(actions) == self.n_envs
        for conn, offset, n_envs in zip(
                self.conns, self.worker_env_offsets, self.n_worker_envs):
            conn.send(('step', actions[offset:offset + n_envs]))

        results = []
        for conn in self.conns:
            results.extend(conn.recv())
        next_obs, rews, dones, infos = zip(*results)
        self.last_obs = next_obs = np.asarray(next_obs, dtype=np.float32)
        return next_obs, rews, dones, infos

    def seed(self, seeds):
        assert len(seeds) == self.n_envs
        for conn, offset, n_envs in zip(
                self.conns, self.worker_env_offsets, self.n_worker_envs):
            conn.send(('seed', seeds[offset:offset + n_envs]))

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        for conn in self.conns:
            conn.send(('close', None))
        for worker in self.workers:
            worker.join()
        self.workers = []
        self.conns = []
