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
        for env, (last_done, done) in enumerate(zip(last_dones, dones)):
            if done:
                slices.append((slice(last_done, step+1), env))
                last_dones[env] = step
        obs = next_obs
    env_pool.flush()

    for env, last_done in enumerate(last_dones):
        slices.append((slice(last_done, None), env))
    all_obs[-1] = obs

    return dict(
        observations=all_obs,
        actions=all_acts,
        rewards=all_rews,
        dones=all_dones,
        slices=slices
    )


def samples_generator(env_pool, policy, k, compute_dists_vals):
    obs = env_pool.reset()
    dists, vals = compute_dists_vals(torch.as_tensor(obs))

    n = env_pool.n_envs
    while True:
        all_acts = torch.empty((k, n) + policy.pdtype.sample_shape)
        all_dists = torch.empty((k, n) + policy.pdtype.param_shape)
        all_rews = np.empty((k, n), dtype=np.float32)
        all_dones = np.empty((k, n), dtype=np.float32)
        all_vals = torch.empty((k, n))

        for i in range(k):
            with torch.no_grad():
                acts = dists.sample()

            next_obs, rews, dones, _ = env_pool.step(acts.numpy())
            all_acts[i] = acts
            all_rews[i] = rews
            all_dones[i] = dones
            all_dists[i] = dists.flat_params
            all_vals[i] = vals

            dists, vals = compute_dists_vals(torch.as_tensor(next_obs))

        all_rews = torch.from_numpy(all_rews)
        all_dones = torch.from_numpy(all_dones)
        all_dists = policy.pdtype.from_flat(all_dists.reshape(k*n, -1))
        yield all_acts, all_rews, all_dones, all_dists, all_vals, vals.detach()

# ==============================
# Variables and estimation
# ==============================

@torch.no_grad()
def compute_pg_vars(buffer, policy, baseline, gamma, gaelam):
    """
    Compute variables needed for various policy gradient algorithms
    """
    observations = torch.from_numpy(buffer.pop("observations"))
    actions = buffer["actions"]
    rewards = buffer["rewards"]
    dones = buffer.pop("dones")
    returns = rewards.copy()
    baselines = torch.empty(observations.shape[:2])

    slices = buffer.pop("slices")
    for interval in slices:
        baselines[interval] = baseline(observations[interval])
    # baselines = baseline(observations)
    values = baselines.numpy()
    values[-1, dones[-1]] = 0
    deltas = rewards + gamma*values[1:] - values[:-1]
    returns[-1] += gamma * values[-1]

    gaemul = gamma * gaelam
    for step in reversed(range(len(rewards)-1)):
        deltas[step] += (1-dones[step]) * gaemul * deltas[step+1]
        returns[step] += (1-dones[step]) * gamma * returns[step+1]

    # Normalizing the advantage values can make the algorithm more robust to
    # reward scaling
    buffer["advantages"] = (deltas-deltas.mean()) / deltas.std()
    buffer["observations"] = observations[:-1]
    buffer["baselines"] = baselines[:-1]
    buffer["returns"] = returns

    batch_size = np.prod(rewards.shape)
    for k, v in buffer.items():
        buffer[k] = torch.as_tensor(v).reshape(batch_size, -1).squeeze()

    return buffer['observations'], buffer['actions'], buffer['advantages']
