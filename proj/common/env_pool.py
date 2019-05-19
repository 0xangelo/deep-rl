import numpy as np
import multiprocessing as mp
import ctypes
from baselines.common.vec_env import VecEnv

# ==============================
# Using mp.Pipe (pickles data)
# ==============================

def env_worker(env_maker, conn, n_envs):
    envs = [env_maker() for _ in range(n_envs)]
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
            elif command == 'get_spaces':
                conn.send((envs[0].observation_space, envs[0].action_space))
            elif command == 'render':
                conn.send([env.render(mode='rgb_array') for env in envs])
            elif command == 'close':
                break
            else:
                raise ValueError("Unrecognized command: {}".format(command))
    except KeyboardInterrupt:
        print('EnvPool worker: got KeyboardInterrupt')
    finally:
        conn.close()
        for env in envs:
            env.close()


class EnvPool(VecEnv):
    """
    Using a pool of workers to run multiple environments in parallel. This
    implementation supports multiple environments per worker to be as flexible
    as possible.
    """

    def __init__(self, env_maker, n_envs=mp.cpu_count(),
                 n_parallel=mp.cpu_count() / 2):
        # No point in having more parallel workers than environments
        self.n_parallel = n_envs if n_parallel > n_envs else n_parallel
        # try to split evenly, but this isn't always possible
        num_worker_envs = [
            len(d) for d in np.array_split(np.arange(n_envs), self.n_parallel)]
        self.worker_env_seps = np.concatenate([[0], np.cumsum(num_worker_envs)])

        self.workers, self.conns = [], []
        for num_envs in (self.worker_env_seps[1:] - self.worker_env_seps[:-1]):
            worker_conn, master_conn = mp.Pipe()
            worker = mp.Process(
                target=env_worker, args=(env_maker, worker_conn, num_envs))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
            self.conns.append(master_conn)

        self.conns[0].send(('get_spaces', None))
        ob_space, ac_space = self.conns[0].recv()
        super().__init__(n_envs, ob_space, ac_space)

        self.waiting = False
        self.closed = False

        # set initial seeds
        seeds = np.random.randint(
            low=0, high=np.iinfo(np.int32).max, size=n_envs)
        self.seed(seeds)

    def reset(self):
        assert not self.closed
        if self.waiting:
            self.step_wait()
        for conn in self.conns:
            conn.send(('reset', None))
        obs = []
        for conn in self.conns:
            obs.extend(conn.recv())
        return np.stack(obs)

    def step_async(self, actions):
        assert not self.waiting and not self.closed
        for conn, acts in zip(self.conns,
                              np.split(actions, self.worker_env_seps[1:-1])):
            conn.send(('step', acts))
        self.waiting = True

    def step_wait(self):
        assert self.waiting and not self.closed
        results = []
        for conn in self.conns:
            results.extend(conn.recv())
        next_obs, rews, dones, infos = zip(*results)
        self.waiting = False
        return np.stack(next_obs), np.stack(rews), np.stack(dones), infos

    def seed(self, seeds):
        assert not self.waiting and not self.closed
        for conn, data in zip(self.conns,
                              np.split(seeds, self.worker_env_seps[1:-1])):
            conn.send(('seed', data))

    def close_extras(self):
        if self.waiting:
            self.step_wait()
        for conn in self.conns:
            conn.send(('close', None))
            conn.close()
        for worker in self.workers:
            worker.join()

    def get_images(self):
        assert not self.waiting and not self.closed
        for conn in self.conns:
            conn.send(('render', None))
        imgs = []
        for conn in self.conns:
            imgs.extend(conn.recv())
        return imgs

# ==============================
# Using shared memory
# ==============================

_NP_TO_CT = {np.float32: ctypes.c_float,
             np.int32: ctypes.c_int32,
             np.int8: ctypes.c_int8,
             np.uint8: ctypes.c_char,
             np.bool: ctypes.c_bool}


def shm_worker(env_maker, conn, n_envs, obs_bufs, obs_shape, obs_dtype):
    envs = [env_maker() for _ in range(n_envs)]
    def _write_obs(obs):
        for ob, obs_buf in zip(obs, obs_bufs):
            dst = obs_buf.get_obj()
            dst_np = np.frombuffer(dst, dtype=obs_dtype).reshape(obs_shape)  # pylint: disable=W0212
            np.copyto(dst_np, ob)
    try:
        while True:
            command, data = conn.recv()
            if command == 'reset':
                conn.send(_write_obs([env.reset() for env in envs]))
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
            elif command == 'render':
                conn.send([env.render(mode='rgb_array') for env in envs])
            elif command == 'close':
                break
            else:
                raise RuntimeError("Unrecognized command: {}".format(command))
    except KeyboardInterrupt:
        print('ShmEnvPool worker: got KeyboardInterrupt')
    finally:
        conn.close()
        for env in envs:
            env.close()


class ShmEnvPool(VecEnv):
    def __init__(self, env_maker, n_envs=mp.cpu_count(),
                 n_parallel=mp.cpu_count() / 2):
        dummy = env_maker()
        ob_space, ac_space = dummy.observation_space, dummy.action_space
        del dummy
        super().__init__(n_envs, ob_space, ac_space)

        # No point in having more parallel workers than environments
        self.n_parallel = n_envs if n_parallel > n_envs else n_parallel
        # try to split evenly, but this isn't always possible
        num_worker_envs = [
            len(d) for d in np.array_split(np.arange(n_envs), self.n_parallel)]
        self.worker_env_seps = np.concatenate([[0], np.cumsum(num_worker_envs)])

        self.obs_dtype, self.obs_shape = ob_space.dtype, ob_space.shape
        self.obs_bufs = []
        for _ in range(n_envs):
            self.obs_bufs.append(mp.Array(_NP_TO_CT[ob_space.dtype.type],
                                          int(np.prod(ob_space.shape))))

        self.workers, self.conns = [], []
        for beg, end in zip(self.worker_env_seps[:-1],
                            self.worker_env_seps[1:]):
            worker_conn, master_conn = mp.Pipe()
            worker = mp.Process(
                target=shm_worker,
                args=(env_maker, worker_conn, end - beg,
                      self.obs_bufs[beg:end], ob_space.shape, ob_space.dtype))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
            self.conns.append(master_conn)

        self.waiting = False
        self.closed = False

        # set initial seeds
        seeds = np.random.randint(
            low=0, high=np.iinfo(np.int32).max, size=n_envs)
        self.seed(seeds)

    def reset(self):
        assert not self.closed
        if self.waiting:
            self.step_wait()
        for conn in self.conns:
            conn.send(('reset', None))
        for conn in self.conns:
            conn.recv()
        return self._decode_obses()

    def step_async(self, actions):
        assert not self.waiting and not self.closed
        for conn, acts in zip(self.conns,
                              np.split(actions, self.worker_env_seps[1:-1])):
            conn.send(('step', acts))
        self.waiting = True

    def step_wait(self):
        assert self.waiting and not self.closed
        results = []
        for conn in self.conns:
            results.extend(conn.recv())
        rews, dones, infos = zip(*results)
        self.waiting = False
        return self._decode_obses(), np.stack(rews), np.stack(dones), infos

    def seed(self, seeds):
        assert not self.waiting and not self.closed
        for conn, data in zip(self.conns,
                              np.split(seeds, self.worker_env_seps[1:-1])):
            conn.send(('seed', data))

    def close_extras(self):
        if self.waiting:
            self.step_wait()
        for conn in self.conns:
            conn.send(('close', None))
            conn.close()
        for worker in self.workers:
            worker.join()
        self.obs_bufs.clear()

    def get_images(self):
        assert not self.waiting and not self.closed
        for conn in self.conns:
            conn.send(('render', None))
        imgs = []
        for conn in self.conns:
            imgs.extend(conn.recv())
        return imgs

    def _decode_obses(self):
        results = [
            np.frombuffer(b.get_obj(), dtype=self.obs_dtype).reshape(
                self.obs_shape)
            for b in self.obs_bufs]
        return np.stack(results)
