import gym, torch, numpy as np, multiprocessing as mp
from . import logger
from .utils import explained_variance_1d, kl
from .tqdm_util import trange
from .env_pool import EnvPool, parallel_collect_experience
from .distributions import DiagNormal, Categorical


# ==============================
# Variables and estimation
# ==============================

def compute_cumulative_returns(rewards, baselines, discount):
    """
    Builds up the cumulative sum of discounted rewards for each time step:
    R[t] = sum_{t'>=t} γ^(t'-t)*r_t'
    Note that we use γ^(t'-t) instead of γ^t'. This gives us a biased gradient 
    but lower variance.
    """
    rews = rewards.cpu()
    returns = torch.empty_like(rews)
    # Use the last baseline prediction to back up
    cum_return = baselines[-1].cpu()
    for idx in reversed(range(len(rews))):
        returns[idx] = cum_return = cum_return * discount + rews[idx]
    return returns.to(rewards)


# def compute_advantages(rewards, baselines, discount, *args):
#     """
#     Given returns R_t and baselines b(s_t), compute (monte carlo) advantage
#     estimate A_t.
#     """
#     returns = compute_cumulative_returns(rewards, baselines, discount)
#     gt = discount ** torch.arange(len(returns), dtype=torch.get_default_dtype())
#     return gt * (returns - baselines[:-1])


def compute_advantages(rewards, baselines, discount, gae_lambda):
    """
    Given returns R_t and baselines b(s_t), compute (generalized) advantage 
    estimate A_t.
    """
    deltas = (rewards + discount * baselines[1:] - baselines[:-1]).cpu()
    advs = torch.empty_like(deltas)
    cum_adv = 0
    multiplier = discount * gae_lambda
    for idx in reversed(range(len(deltas))):
        advs[idx] = cum_adv = cum_adv * multiplier + deltas[idx]
    return advs.to(rewards)


@torch.no_grad()
def compute_pg_vars(trajs, policy, baseline, discount, gae_lambda):
    """
    Compute variables needed for various policy gradient algorithms
    """
    for traj in trajs:
        # Include the last observation here, if the trajectory is not finished
        baselines = baseline(torch.cat(
            [traj["observations"], traj["last_observation"].unsqueeze(0)]))
        if traj['finished']:
            # If already finished, the future cumulative rewards starting from
            # the final state is 0
            baselines[-1].zero_()
        # This is useful when fitting baselines. It uses the baseline prediction
        # of the last state value to perform Bellman backup if the trajectory is
        # not finished.
        traj['returns'] = compute_cumulative_returns(
            traj['rewards'], baselines, discount)
        traj['advantages'] = compute_advantages(
            traj['rewards'], baselines, discount, gae_lambda)
        traj['baselines'] = baselines[:-1]

    # First, we compute a flattened list of observations, actions, advantages
    all_obs = torch.cat([traj['observations'] for traj in trajs])
    all_acts = torch.cat([traj['actions'] for traj in trajs])
    all_advs = torch.cat([traj['advantages'] for traj in trajs])
    all_dists = policy.dists(all_obs)

    # Normalizing the advantage values can make the algorithm more robust to
    # reward scaling
    all_advs = (all_advs) / (all_advs.std() + 1e-8)

    return all_obs, all_acts, all_advs, all_dists


# ==============================
# Helper methods for logging
# ==============================

def log_reward_statistics(env):
    # keep unwrapping until we get the monitor
    while not isinstance(env, gym.wrappers.Monitor):
        if not isinstance(env, gym.Wrapper):
            assert False
        env = env.env
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
    baselines = torch.cat([traj['baselines'] for traj in trajs])
    returns = torch.cat([traj['returns'] for traj in trajs])
    logger.logkv('ExplainedVariance', explained_variance_1d(baselines, returns))


@torch.no_grad()
def log_action_distribution_statistics(dists, policy, obs):
    logger.logkv('MeanKL', kl(dists, policy.dists(obs)).mean().item())
    logger.logkv('Entropy', dists.entropy().mean().item())
    logger.logkv('Perplexity', dists.perplexity().mean().item())
    if isinstance(dists, DiagNormal):
        logger.logkv('AveragePolicyStd', dists.stddev.mean().item())
        for idx in range(dists.stddev.shape[-1]):
            logger.logkv('AveragePolicyStd[{}]'.format(idx),
                          dists.stddev[...,idx].mean().item())
    elif isinstance(dists, Categorical):
        probs = dists.probs.mean(0)
        for idx in range(probs.numel()):
            logger.logkv('AveragePolicyProb[{}]'.format(idx), probs[idx].item())

