import gym, torch, numpy as np
from torch.distributions.kl import kl_divergence as kl
from proj.utils import logger
from proj.common.utils import explained_variance_1d
from proj.common.distributions import DiagNormal, Categorical


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


def log_baseline_statistics(buffer):
    # Specifically, compute the explained variance, defined as
    baselines = buffer['baselines']
    returns = buffer['returns']
    logger.logkv('ExplainedVariance', explained_variance_1d(baselines, returns))


@torch.no_grad()
def log_action_distribution_statistics(dists):
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

@torch.no_grad()
def log_average_kl_divergence(old_dists, policy, obs):
    logger.logkv('MeanKL', kl(old_dists, policy.dists(obs)).mean().item())
