import numpy as np
import multiprocessing as mp
from gym.wrappers import Monitor
import ctypes


_NP_TO_CT = {np.float32: ctypes.c_float,
             np.int32: ctypes.c_int32,
             np.int8: ctypes.c_int8,
             np.uint8: ctypes.c_char,
             np.bool: ctypes.c_bool}

# ==============================
# Parallel traj collecting
# ==============================

def env_worker(env_maker, conn, n_envs):
    envs = [env_maker() for _ in range(n_envs)]
    flushers = []
    for env in envs:
        while not isinstance(env, Monitor):
            env = env.env
        flushers.append(env._flush)
    try:
        while True:
            command, data = conn.recv()
            if command == 'reset':
                conn.send([env.reset() for env in envs])
            elif command == 'seed':
                for env, seed in zip(envs, data):
                    env.seed(int(seed))
            elif command == 'step':
                results = []
                for env, action in zip(envs, data):
                    next_ob, rew, done, info = env.step(action)
                    if done: next_ob = env.reset()
                    results.append((next_ob, rew, done, info))
                conn.send(results)
            elif command == 'flush':
                for flusher in flushers:
                    flusher(True)
                conn.send(None)
            elif command == 'close':
                conn.close()
                break
            else:
                raise ValueError("Unrecognized command: {}".format(command))
    except KeyboardInterrupt:
        print('EnvPool worker: got KeyboardInterrupt')
    finally:
        for env in envs:
            env.close()


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
        n_worker_envs = [
            len(d) for d in np.array_split(np.arange(n_envs), n_parallel)]
        self.worker_env_seps = np.concatenate([[0], np.cumsum(n_worker_envs)])
        self.last_obs = None

    def start(self):
        for n_envs in (self.worker_env_seps[1:] - self.worker_env_seps[:-1]):
            worker_conn, master_conn = mp.Pipe()
            worker = mp.Process(
                target=env_worker, args=(self.env_maker, worker_conn, n_envs))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
            self.conns.append(master_conn)

        # set initial seeds
        seeds = np.random.randint(
            low=0, high=np.iinfo(np.int32).max, size=self.n_envs)
        self.seed(seeds)

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
        self.last_obs = np.asarray(obs, dtype='f')
        return self.last_obs

    def flush(self):
        for conn in self.conns:
            conn.send(('flush', None))
        for conn in self.conns:
            conn.recv()

    def step(self, actions):
        assert len(actions) == self.n_envs
        for conn, acts in zip(self.conns,
                              np.split(actions, self.worker_env_seps[1:-1])):
            conn.send(('step', acts))

        results = []
        for conn in self.conns:
            results.extend(conn.recv())
        next_obs, rews, dones, infos = zip(*results)
        self.last_obs = np.asarray(next_obs, dtype='f')
        return self.last_obs, np.asarray(rews), np.asarray(dones), infos

    def seed(self, seeds):
        assert len(seeds) == self.n_envs
        for conn, data in zip(self.conns,
                              np.split(seeds, self.worker_env_seps[1:-1])):
            conn.send(('seed', data))

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        for conn in self.conns:
            conn.send(('close', None))
            conn.close()
        for worker in self.workers:
            worker.join()
        self.workers.clear()
        self.conns.clear()


# ==============================
# Parallel traj collecting
# ==============================

def shm_worker(env_maker, conn, n_envs, obs_bufs, obs_shape, obs_dtype):
    envs = [env_maker() for _ in range(n_envs)]
    flushers = []
    for env in envs:
        while not isinstance(env, Monitor):
            env = env.env
        flushers.append(env._flush)
    def _write_obs(obs):
        for ob, obs_buf in zip(obs, obs_bufs):
            dst = obs_buf.get_obj()
            dst_np = np.frombuffer(dst, dtype=obs_dtype).reshape(obs_shape)  # pylint: disable=W0212
            np.copyto(dst_np, ob)
    try:
        while True:
            command, data = conn.recv()
            if command == 'reset':
                _write_obs([env.reset() for env in envs])
                conn.send(None)
            elif command == 'seed':
                for env, seed in zip(envs, data):
                    env.seed(int(seed))
            elif command == 'step':
                results, obs = [], []
                for env, action in zip(envs, data):
                    ob, rew, done, info = env.step(action)
                    if done:
                        ob = env.reset()
                    results.append((rew, done, info))
                    obs.append(ob)
                _write_obs(obs)
                conn.send(results)
            elif command == 'flush':
                for flusher in flushers:
                    flusher(True)
                conn.send(None)
            elif command == 'close':
                conn.close()
                break
            else:
                raise RuntimeError("Unrecognized command: {}".format(command))
    except KeyboardInterrupt:
        print('ShmEnvPool worker: got KeyboardInterrupt')
    finally:
        for env in envs:
            env.close()


class ShmEnvPool(object):
    def __init__(self, env_maker, n_envs=mp.cpu_count(),
                 n_parallel=mp.cpu_count()):
        dummy = env_maker()
        ob_space = dummy.observation_space
        self.obs_shape, self.obs_dtype = ob_space.shape, ob_space.dtype
        del dummy

        self.env_maker = env_maker
        self.n_envs = n_envs
        self.workers = []
        self.conns = []

        # No point in having more parallel workers than environments
        if n_parallel > n_envs:
            n_parallel = n_envs
        self.n_parallel = n_parallel
        # try to split evenly, but this isn't always possible
        n_worker_envs = [
            len(d) for d in np.array_split(np.arange(n_envs), n_parallel)]
        self.worker_env_seps = np.concatenate([[0], np.cumsum(n_worker_envs)])
        self.obs_bufs = []
        self.last_obs = None

    def start(self):
        for _ in range(self.n_envs):
            self.obs_bufs.append(mp.Array(_NP_TO_CT[self.obs_dtype.type],
                                          int(np.prod(self.obs_shape))))
        for beg, end in zip(self.worker_env_seps[:-1],
                            self.worker_env_seps[1:]):
            worker_conn, master_conn = mp.Pipe()
            worker = mp.Process(
                target=shm_worker,
                args=(self.env_maker, worker_conn, end - beg,
                      self.obs_bufs[beg:end], self.obs_shape, self.obs_dtype))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
            self.conns.append(master_conn)

        # set initial seeds
        seeds = np.random.randint(
            low=0, high=np.iinfo(np.int32).max, size=self.n_envs)
        self.seed(seeds)

    def __enter__(self):
        self.start()
        return self

    def seed(self, seeds):
        assert len(seeds) == self.n_envs
        for conn, data in zip(self.conns,
                              np.split(seeds, self.worker_env_seps[1:-1])):
            conn.send(('seed', data))

    def reset(self):
        for conn in self.conns:
            conn.send(('reset', None))
        for conn in self.conns:
            conn.recv()
        self.last_obs = self._decode_obses()
        return self.last_obs

    def step(self, actions):
        assert len(actions) == self.n_envs
        for conn, acts in zip(self.conns,
                              np.split(actions, self.worker_env_seps[1:-1])):
            conn.send(('step', acts))

        results = []
        for conn in self.conns:
            results.extend(conn.recv())
        rews, dones, infos = zip(*results)
        self.last_obs = self._decode_obses()
        return self.last_obs, np.asarray(rews), np.asarray(dones), infos

    def flush(self):
        for conn in self.conns:
            conn.send(('flush', None))
        for conn in self.conns:
            conn.recv()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close(exc_type)

    def close(self, exc_type):
        for conn in self.conns:
            if exc_type is None:
                conn.send(('close', None))
            conn.close()
        for worker in self.workers:
            worker.join()
        self.obs_bufs.clear()
        self.conns.clear()
        self.workers.clear()

    def _decode_obses(self):
        results = [
            np.frombuffer(b.get_obj(), dtype=self.obs_dtype).reshape(
                self.obs_shape)
            for b in self.obs_bufs]
        return np.asarray(results, dtype='f')
