from proj.common.utils import *
from proj.common.distributions import Normal, Categorical

# ==============================
# Shared utilities
# ==============================

def compute_cumulative_returns(rewards, baselines, discount):
    # This method builds up the cumulative sum of discounted rewards for each time step:
    # R[t] = sum_{t'>=t} γ^(t'-t)*r_t'
    # Note that we use γ^(t'-t) instead of γ^t'. This gives us a biased gradient but lower variance
    returns = np.empty_like(rewards)
    # Use the last baseline prediction to back up
    cum_return = baselines[-1]
    for idx in range(len(rewards)-1,-1,-1):
        returns[idx] = cum_return = cum_return * discount + rewards[idx]
    return returns


def compute_advantages(rewards, baselines, discount, gae_lambda):
    # Given returns R_t and baselines b(s_t), compute (generalized) advantage estimate A_t
    deltas = rewards + discount * baselines[1:] - baselines[:-1]
    advs = np.empty_like(deltas)
    cum_adv = 0
    multiplier = discount * gae_lambda
    for idx in range(len(deltas)-1,-1,-1):
        advs[idx] = cum_adv = cum_adv * multiplier + deltas[idx]
    return advs


def compute_pg_vars(trajs, policy, baseline, discount, gae_lambda):
    """
    Compute variables needed for various policy gradient algorithms
    """
    for traj in trajs:
        # Include the last observation here, in case the trajectory is not finished
        baselines = baseline.predict(np.concatenate(
            [traj["observations"], [traj["last_observation"]]]))
        if traj['finished']:
            # If already finished, the future cumulative rewards starting from the final state is 0
            baselines[-1] = 0.
        # This is useful when fitting baselines. It uses the baseline prediction of the last state value to perform
        # Bellman backup if the trajectory is not finished.
        traj['returns'] = compute_cumulative_returns(
            traj['rewards'], baselines, discount)
        traj['advantages'] = compute_advantages(
            traj['rewards'], baselines, discount, gae_lambda)
        traj['baselines'] = baselines[:-1]

    # First, we compute a flattened list of observations, actions, and advantages
    all_obs = np.concatenate([traj['observations'] for traj in trajs], axis=0)
    all_acts = np.concatenate([traj['actions'] for traj in trajs], axis=0)
    all_advs = np.concatenate([traj['advantages'] for traj in trajs], axis=0)
    all_dists = np.concatenate([traj['distributions'] for traj in trajs], axis=0)

    # Normalizing the advantage values can make the algorithm more robust to reward scaling
    all_advs = (all_advs - np.mean(all_advs)) / (np.std(all_advs) + 1e-8)

    all_obs = policy.process_obs(all_obs)
    all_acts = torch.as_tensor(all_acts)
    all_advs = torch.as_tensor(all_advs)
    all_dists = policy.distribution.fromflat(torch.as_tensor(all_dists))

    return all_obs, all_acts, all_advs, all_dists


# ==============================
# Helper methods for logging
# ==============================

def log_reward_statistics(env):
    # keep unwrapping until we get the monitor
    while not isinstance(env, gym.wrappers.Monitor):  # and not isinstance()
        if not isinstance(env, gym.Wrapper):
            assert False
        env = env.env
    # env.unwrapped
    assert isinstance(env, gym.wrappers.Monitor)
    all_stats = None
    for _ in range(10):
        try:
            all_stats = gym.wrappers.monitor.load_results(env.directory)
        except FileNotFoundError:
            time.sleep(1)
            continue
    if all_stats is not None:
        episode_rewards = all_stats['episode_rewards']
        episode_lengths = all_stats['episode_lengths']

        recent_episode_rewards = episode_rewards[-100:]
        recent_episode_lengths = episode_lengths[-100:]

        if len(recent_episode_rewards) > 0:
            logger.logkv('AverageReturn', np.mean(recent_episode_rewards))
            logger.logkv('MinReturn', np.min(recent_episode_rewards))
            logger.logkv('MaxReturn', np.max(recent_episode_rewards))
            logger.logkv('StdReturn', np.std(recent_episode_rewards))
            logger.logkv('AverageEpisodeLength',
                         np.mean(recent_episode_lengths))
            logger.logkv('MinEpisodeLength', np.min(recent_episode_lengths))
            logger.logkv('MaxEpisodeLength', np.max(recent_episode_lengths))
            logger.logkv('StdEpisodeLength', np.std(recent_episode_lengths))

        logger.logkv('TotalNEpisodes', len(episode_rewards))
        logger.logkv('TotalNSamples', np.sum(episode_lengths))


def log_baseline_statistics(trajs):
    # Specifically, compute the explained variance, defined as
    baselines = np.concatenate([traj['baselines'] for traj in trajs])
    returns = np.concatenate([traj['returns'] for traj in trajs])
    logger.logkv('ExplainedVariance',
                 explained_variance_1d(baselines, returns))


@torch.no_grad()
def log_action_distribution_statistics(dists):
    logger.logkv('Entropy', torch.mean(dists.entropy()).numpy())
    logger.logkv('Perplexity', torch.mean(dists.perplexity()).numpy())
    if isinstance(dists, Normal):
        logger.logkv('AveragePolicyStd', torch.mean(dists.stddev).numpy())
        for idx in range(dists.stddev.shape[-1]):
            logger.logkv('AveragePolicyStd[{}]'.format(idx),
                          torch.mean(dists.stddev[...,idx]).numpy())
    elif isinstance(dists, Categorical):
        probs = torch.mean(torch.softmax(dists.logits, dim=1), dim=0).numpy()
        for idx in range(probs.size):
            logger.logkv('AveragePolicyProb[{}]'.format(idx), probs[idx])

