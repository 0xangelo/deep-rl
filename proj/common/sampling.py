import torch
import numpy as np
from proj.utils.tqdm_util import trange
from proj.common.utils import discount_cumsum


@torch.no_grad()
def parallel_collect_samples(env_pool, policy, num_samples):
    """
    Collect trajectories in parallel using a pool of workers. Actions are
    computed using the provided policy. For each worker, \lfloor num_samples /
    env_pool.n_workers \rfloor timesteps are sampled. This means that some of
    the trajectories will not be executed until termination.

    When starting, it will first check if env_pool.last_obs is set, and if so,
    it will start from there rather than resetting all environments. This is
    useful for reusing the same episode.

    :param env_pool: An instance of EnvPool.
    :param policy: The policy used to select actions.
    :param num_samples: The approximate total number of samples to collect.
    :return: A dictionary with all observations, actions, rewards and tuples
    of last index, finished flag and last observation of each trajectory
    """
    n_envs = env_pool.n_envs
    steps = num_samples // n_envs
    all_obs = np.empty(
        (steps+1, n_envs) + policy.ob_space.shape, policy.ob_space.dtype)
    all_acts = np.empty(
        (steps, n_envs) + policy.ac_space.shape, policy.ac_space.dtype)
    all_rews = np.empty((steps, n_envs), dtype=np.float32)
    all_dones = np.empty((steps, n_envs), dtype=np.bool)
    slices = []
    last_dones = [0] * n_envs

    obs = env_pool.reset() if env_pool.last_obs is None else env_pool.last_obs
    for step in trange(steps, unit="step", leave=False, desc="Sampling"):
        actions = policy.actions(torch.from_numpy(obs)).numpy()
        next_obs, rews, dones, _ = env_pool.step(actions)
        all_obs[step] = obs
        all_acts[step] = actions
        all_rews[step] = rews
        all_dones[step] = dones
        obs = next_obs
    env_pool.flush()

    for env, last_done in enumerate(last_dones):
        slices.append((slice(last_done, None), env))
    all_obs[-1] = obs

    return dict(
        observations=all_obs,
        actions=all_acts,
        rewards=all_rews,
        dones=all_dones
    )


def samples_generator(env_pool, policy, k, compute_dists_vals):
    obs = env_pool.reset()
    dists, vals = compute_dists_vals(torch.from_numpy(obs))

    n = env_pool.n_envs
    while True:
        all_acts = torch.empty((k, n) + policy.pdtype.sample_shape)
        all_dists = torch.empty((k, n) + policy.pdtype.param_shape)
        all_rews = torch.empty((k, n))
        all_dones = torch.empty((k, n))
        all_vals = torch.empty((k, n))

        for i in range(k):
            with torch.no_grad():
                acts = dists.sample()

            next_obs, rews, dones, _ = env_pool.step(acts.numpy())
            all_acts[i] = acts
            all_rews[i] = torch.from_numpy(rews.astype('f'))
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
def compute_pg_vars(buffer, policy, val_fn, gamma, gaelam):
    """
    Compute variables needed for various policy gradient algorithms
    """
    observations = buffer.pop("observations")
    rewards = buffer.pop("rewards")
    masks = np.invert(buffer.pop("dones"))

    values = val_fn(torch.from_numpy(observations)).numpy()
    deltas = rewards + gamma * (masks*values[1:]) - values[:-1]
    rewards[-1] += gamma * (masks[-1]*values[-1])
    gaemul = gamma * gaelam
    for step in reversed(range(len(rewards)-1)):
        deltas[step] += gaemul * (masks[step]*deltas[step+1])
        rewards[step] += gamma * (masks[step]*rewards[step+1])

    # Normalizing the advantage values can make the algorithm more robust to
    # reward scaling
    buffer["advantages"] = (deltas-deltas.mean()) / deltas.std()
    buffer["observations"] = observations[:-1]
    buffer["values"] = values[:-1]
    buffer["returns"] = rewards

    batch_size = np.prod(rewards.shape)
    for k, v in buffer.items():
        buffer[k] = torch.from_numpy(v).reshape(batch_size, -1).squeeze()

    return buffer['observations'], buffer['actions'], buffer['advantages']
