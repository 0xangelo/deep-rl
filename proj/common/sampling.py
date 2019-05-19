import torch
import random
from collections import OrderedDict
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from proj.utils.tqdm_util import trange
from proj.utils.torch_util import _NP_TO_PT


class ReplayBuffer(object):
    def __init__(self, capacity, ob_space, ac_space):
        self.all_obs1 = torch.empty(capacity, *ob_space.shape)
        self.all_acts = torch.empty(capacity, *ac_space.shape)
        self.all_rews = torch.empty(capacity)
        self.all_obs2 = torch.empty(capacity, *ob_space.shape)
        self.all_dones = torch.empty(capacity)
        self.ptr, self.size, self.capacity = 0, 0, capacity

    def store(self, ob1, act, rew, ob2, done):
        self.all_obs1[self.ptr] = ob1
        self.all_acts[self.ptr] = act
        self.all_rews[self.ptr] = rew
        self.all_obs2[self.ptr] = ob2
        self.all_dones[self.ptr] = done
        self.ptr = (self.ptr+1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, mb_size):
        idxs = random.sample(range(self.size), mb_size)
        return self.all_obs1[idxs], self.all_acts[idxs], self.all_rews[idxs], \
            self.all_obs2[idxs], self.all_dones[idxs]

    def state_dict(self):
        return self.__dict__.copy()

    def load_state_dict(self, state_dict):
        assert self.capacity == state_dict['capacity'], \
            "Trying to load state between incompatible replay buffers in size"
        self.__dict__.update(state_dict)


@torch.no_grad()
def parallel_samples_collector(vec_env, policy, steps):
    """
    Collect trajectories in parallel using a vectorized environment.
    Actions are computed using the provided policy. For each worker,
    'steps' timesteps are sampled. This means that some of the
    trajectories will not be executed until termination.

    :param vec_env: An instance of baselines.common.vec_env.VecEnv.
    :param policy: An instance of proj.common.models.Policy.
    :param steps: The number of steps to take in each environment.
    :return: An OrderedDict with all observations, actions, rewards
        and done flags as matrixes of size (steps, vec_envs).
    """
    n_envs = vec_env.num_envs
    ob_space, ac_space = vec_env.observation_space, vec_env.action_space

    obs = torch.as_tensor(vec_env.reset(), dtype=_NP_TO_PT[ob_space.dtype.type])
    while True:
        all_obs, all_acts, all_rews, all_dones = [], [], [], []
        for step in trange(steps, unit="step", leave=False, desc="Sampling"):
            actions = policy.actions(obs)
            next_obs, rews, dones, _ = vec_env.step(actions.numpy())
            all_obs.append(obs)
            all_acts.append(actions)
            all_rews.append(torch.as_tensor(rews.astype('f')))
            all_dones.append(torch.from_numpy(dones.astype('f')))
            obs = torch.as_tensor(next_obs, dtype=_NP_TO_PT[ob_space.dtype.type])

        all_obs.append(obs)
        yield OrderedDict(
            observations=torch.stack(all_obs),
            actions=torch.stack(all_acts),
            rewards=torch.stack(all_rews),
            dones=torch.stack(all_dones))


# @torch.no_grad()
# def parallel_samples_collector(vec_env, policy, steps):
#     """
#     Collect trajectories in parallel using a vectorized environment.
#     Actions are computed using the provided policy. For each worker,
#     'steps' timesteps are sampled. This means that some of the
#     trajectories will not be executed until termination.

#     :param vec_env: An instance of baselines.common.vec_env.VecEnv.
#     :param policy: An instance of proj.common.models.Policy.
#     :param steps: The number of steps to take in each environment.
#     :return: An OrderedDict with all observations, actions, rewards
#         and done flags as matrixes of size (steps, vec_envs).
#     """
#     n_envs = vec_env.num_envs
#     ob_space, ac_space = vec_env.observation_space, vec_env.action_space

#     obs = torch.from_numpy(vec_env.reset())
#     while True:
#         all_obs = torch.empty((steps+1, n_envs) + ob_space.shape,
#                               dtype=_NP_TO_PT[ob_space.dtype.type])
#         all_acts = torch.empty((steps, n_envs) + ac_space.shape,
#                                dtype=_NP_TO_PT[ac_space.dtype.type])
#         all_rews = torch.empty((steps, n_envs))
#         all_dones = torch.empty((steps, n_envs), dtype=torch.uint8)

#         for step in trange(steps, unit="step", leave=False, desc="Sampling"):
#             actions = policy.actions(obs)
#             next_obs, rews, dones, _ = vec_env.step(actions.numpy())
#             all_obs[step] = obs
#             all_acts[step] = actions
#             all_rews[step] = torch.from_numpy(rews)
#             all_dones[step] = torch.from_numpy(dones.astype('f'))
#             obs = torch.from_numpy(next_obs)

#         all_obs[-1] = obs
#         yield OrderedDict(observations=all_obs, actions=all_acts,
#                           rewards=all_rews, dones=all_dones)


def samples_generator(vec_env, policy, k, compute_dists_vals):
    obs = vec_env.reset()
    dists, vals = compute_dists_vals(torch.from_numpy(obs))

    n = vec_env.num_envs
    while True:
        all_acts = torch.empty((k, n) + policy.pdtype.sample_shape)
        all_dists = torch.empty((k, n) + policy.pdtype.param_shape)
        all_rews = torch.empty((k, n))
        all_dones = torch.empty((k, n))
        all_vals = torch.empty((k, n))

        for i in range(k):
            with torch.no_grad():
                acts = dists.sample()

            next_obs, rews, dones, _ = vec_env.step(acts.numpy())
            all_acts[i] = acts
            all_rews[i] = torch.from_numpy(rews)
            all_dones[i] = torch.from_numpy(dones.astype('f'))
            all_dists[i] = dists.flat_params
            all_vals[i] = vals

            dists, vals = compute_dists_vals(torch.from_numpy(next_obs))

        all_dists = policy.pdtype.from_flat(all_dists.reshape(k*n, -1))
        yield all_acts, all_rews, all_dones, all_dists, all_vals, vals.detach()

# ==============================
# Variables and estimation
# ==============================

@torch.no_grad()
def compute_pg_vars(trajs, policy, val_fn, gamma, gaelam):
    """
    Compute variables needed for various policy gradient algorithms.
    Adds advantages, values and returns to the provided trajectories.

    :param trajs: An OrderedDict with observations, actions, rewards
        and done flags. Assumes all have the same batch size except
        observations with one more.
    :param policy: An instance of proj.common.models.Policy.
    :param val_fn: An instance of proj.common.models.ValueFunction
    :return: A tuple of all advantages, values and returns computed
    """
    observations, _, rewards, dones = trajs.values()
    masks = (1 - dones).to(rewards)
    returns = rewards.clone()

    # values_shape = torch.cat((rewards, rewards[:1])).shape
    n_steps, n_envs = rewards.shape
    observations = observations.reshape(n_steps+1, n_envs, -1)
    values = val_fn(observations).reshape(n_steps+1, n_envs)
    deltas = rewards + gamma * (masks*values[1:]) - values[:-1]
    returns[-1] += gamma * (masks[-1]*values[-1])
    gaemul = gamma * gaelam
    for step in reversed(range(n_steps-1)):
        deltas[step] += gaemul * (masks[step]*deltas[step+1])
        returns[step] += gamma * (masks[step]*returns[step+1])

    # Normalizing the advantage values can make the algorithm more robust to
    # reward scaling
    advantages = (deltas-deltas.mean()) / deltas.std()

    trajs["advantages"] = advantages
    trajs["values"] = values
    trajs["returns"] = returns


def flatten_trajs(trajs):
    """
    Flattens the entries in trajs along the first dimension.
    """
    for k, v in trajs.items():
        trajs[k] = v.reshape(v.shape[:2].numel(), -1).squeeze()
