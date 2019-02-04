import json
import torch
import os.path
import numpy as np
from baselines import logger
from baselines.bench.monitor import load_results
from torch.distributions.kl import kl_divergence as kl
from proj.utils.json_util import convert_json
from proj.common import distributions
from proj.common.utils import explained_variance_1d


def save_config(config):
    variant_path = os.path.join(logger.get_dir(), 'variant.json')
    params = {}
    if os.path.exists(variant_path):
        with open(variant_path, 'r') as f:
            params = json.load(f)
    with open(variant_path, 'wt') as f:
        json.dump({**params, **convert_json(config)}, f)

# ==============================
# Helper methods for logging
# ==============================

def log_reward_statistics(vec_env):
    all_stats = None
    for _ in range(10):
        try:
            all_stats = load_results(vec_env.directory)
        except FileNotFoundError:
            time.sleep(1)
            continue
    if all_stats is not None:
        episode_rewards = all_stats['r']
        episode_lengths = all_stats['l']

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


@torch.no_grad()
def log_val_fn_statistics(values, returns):
    logger.logkv('ValueLoss', torch.nn.MSELoss()(values, returns).item())
    logger.logkv('ExplainedVariance', explained_variance_1d(values, returns))


@torch.no_grad()
def log_action_distribution_statistics(dists):
    logger.logkv('Entropy', dists.entropy().mean().item())
    logger.logkv('Perplexity', dists.perplexity().mean().item())
    if isinstance(dists, distributions.DiagNormal):
        logger.logkv('AveragePolicyStd', dists.stddev.mean().item())
        for idx in range(dists.stddev.shape[-1]):
            logger.logkv('AveragePolicyStd[{}]'.format(idx),
                          dists.stddev[...,idx].mean().item())
    elif isinstance(dists, distributions.Categorical):
        probs = dists.probs.mean(0).tolist()
        for idx, prob in enumerate(probs):
            logger.logkv('AveragePolicyProb[{}]'.format(idx), prob)


@torch.no_grad()
def log_average_kl_divergence(old_dists, policy, obs):
    logger.logkv('MeanKL', kl(old_dists, policy(obs)).mean().item())
