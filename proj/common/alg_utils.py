import gym, torch, numpy as np, multiprocessing as mp
from torch.distributions.kl import kl_divergence as kl
from . import logger
from .utils import explained_variance_1d, discount_cumsum
from .tqdm_util import trange
from .env_pool import EnvPool, parallel_collect_samples
from .distributions import DiagNormal, Categorical


# ==============================
# Variables and estimation
# ==============================

@torch.no_grad()
def compute_pg_vars(buffer, policy, baseline, gamma, gaelam):
    """
    Compute variables needed for various policy gradient algorithms
    """
    observations = buffer["observations"]
    actions = buffer["actions"]
    rewards = buffer["rewards"]
    returns = buffer["returns"] = np.empty_like(rewards)
    baselines = buffer["baselines"] = baseline(torch.as_tensor(
        observations)).numpy()
    advantages = buffer["advantages"] = np.empty_like(rewards)
    finishes = buffer.pop("finishes")

    beg = 0
    for end, finished, last_obs in sorted(finishes, key=lambda x: x[0]):
        # If already finished, the future cumulative rewards starting from
        # the final state is 0
        value = [0] if finished else baseline(torch.as_tensor(
            last_obs)).numpy()[np.newaxis]
        # This is useful when fitting baselines. It uses the baseline prediction
        # of the last state value to perform Bellman backup if the trajectory is
        # not finished.
        extended_rewards = np.concatenate((rewards[beg:end], value))
        returns[beg:end] = discount_cumsum(extended_rewards, gamma)[:-1]
        values = np.concatenate((baselines[beg:end], value))
        deltas = (rewards[beg:end] + gamma * values[1:] - values[:-1])
        advantages[beg:end] = discount_cumsum(deltas, gamma * gaelam)
        beg = end

    # Normalizing the advantage values can make the algorithm more robust to
    # reward scaling
    buffer["advantages"] = (advantages - advantages.mean()) / advantages.std()

    # Flattened lists of observations, actions, advantages ...
    for key, val in buffer.items():
        buffer[key] = torch.as_tensor(val)
    buffer["distributions"] = policy.dists(buffer["observations"])

    return tuple(
        buffer[key]
        for key in ['observations', 'actions', 'advantages', 'distributions']
    )


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


def log_baseline_statistics(buffer):
    # Specifically, compute the explained variance, defined as
    baselines = buffer['baselines']
    returns = buffer['returns']
    logger.logkv('ExplainedVariance', explained_variance_1d(baselines, returns))


@torch.no_grad()
def log_action_distribution_statistics(buffer, policy):
    dists = buffer["distributions"]
    new_dists = policy.dists(buffer["observations"])
    logger.logkv('MeanKL', kl(dists, new_dists).mean().item())
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

