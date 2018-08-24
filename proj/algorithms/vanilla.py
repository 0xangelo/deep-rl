import torch
from tqdm import trange
from proj.common import logger
from proj.common.alg_utils import *

def vanilla(env, env_maker, policy, baseline, n_iter=100, n_batch=2000, n_envs=mp.cpu_count(),
            optimizer=None, last_iter=-1, gamma=0.99, gae_lambda=0.97, snapshot_saver=None):

    if optimizer is None:
        optimizer = torch.optim.Adam(policy.parameters())

    # Algorithm main loop
    with EnvPool(env_maker, n_envs=n_envs) as env_pool:
        for iter in trange(last_iter + 1, n_iter, desc="Training",
                           unit="updts", file=std_out(), dynamic_ncols=True):
            logger.info("Starting iteration {}".format(iter))
            logger.logkv("Iteration", iter)
            
            logger.info("Start collecting samples")
            trajs = parallel_collect_samples(env_pool, policy, n_batch)
            
            logger.info("Computing policy gradient variables")
            all_obs, all_acts, all_advs, all_dists = compute_pg_vars(
                trajs, policy, baseline, gamma, gae_lambda
            )

            logger.info("Applying policy gradient")
            optimizer.zero_grad()
            surr_loss = -torch.mean(policy(all_obs).log_prob(all_acts) * all_advs)
            surr_loss.backward()
            optimizer.step()

            logger.info("Updating baseline")
            baseline.update(trajs)

            logger.info("Logging information")
            logger.logkv("SurrLoss", surr_loss.item())
            log_reward_statistics(env)
            log_baseline_statistics(trajs)
            log_action_distribution_statistics(all_dists)
            logger.dumpkvs()

            if snapshot_saver is not None:
                logger.info("Saving snapshot")
                snapshot_saver.save_state(
                    iter,
                    dict(
                        alg=vanilla,
                        alg_state=dict(
                            env_maker=env_maker,
                            policy=policy,
                            baseline=baseline,
                            n_iter=n_iter,
                            n_batch=n_batch,
                            n_envs=n_envs,
                            optimizer=optimizer,
                            last_iter=last_iter,
                            gamma=gamma,
                            gae_lambda=gae_lambda
                        )
                    )
                )
