import torch, numpy as np
from proj.utils.tqdm_util import trange
from proj.common.utils import discount_cumsum

@torch.no_grad()
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
    for idx in trange(offset, unit="step", leave=False, desc="Sampling"):
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
    last_infos = tuple(map(list, zip(*sorted(finishes, key=lambda x: x[0]))))

    return dict(
        observations=all_obs,
        actions=all_acts,
        rewards=all_rews,
        last_infos=last_infos
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

        all_rews = torch.as_tensor(all_rews)
        all_dones = torch.as_tensor(all_dones)
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
    observations = buffer["observations"]
    actions      = buffer["actions"]
    rewards      = buffer["rewards"]
    returns      = buffer["returns"] = np.empty_like(rewards)
    baselines    = buffer["baselines"] = baseline(
        torch.as_tensor(observations)).numpy()
    advantages   = buffer["advantages"] = np.empty_like(rewards)

    times, dones, last_obs = buffer.pop("last_infos")
    last_vals = baseline(torch.as_tensor(np.stack(last_obs))).numpy()
    intervals = [slice(*inter) for inter in zip([0] + times[:-1], times)]

    for inter, done, last_val in zip(intervals, dones, last_vals):
        # If already finished, the future cumulative rewards starting from
        # the final state is 0
        last_val = [0] if done else last_val[np.newaxis]
        # This is useful when fitting baselines. It uses the baseline prediction
        # of the last state value to perform Bellman backup if the trajectory is
        # not finished.
        extended_rewards  = np.concatenate((rewards[inter], last_val))
        returns[inter]    = discount_cumsum(extended_rewards, gamma)[:-1]
        values            = np.concatenate((baselines[inter], last_val))
        deltas            = rewards[inter] + gamma * values[1:] - values[:-1]
        advantages[inter] = discount_cumsum(deltas, gamma * gaelam)

    # Normalizing the advantage values can make the algorithm more robust to
    # reward scaling
    buffer["advantages"] = (advantages - advantages.mean()) / advantages.std()

    # Flattened lists of observations, actions, advantages ...
    for key, val in buffer.items():
        buffer[key] = torch.as_tensor(val)

    return buffer['observations'], buffer['actions'], buffer['advantages']
