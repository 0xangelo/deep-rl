from torch.nn.utils import parameters_to_vector, vector_to_parameters
from ..common.utils import flat_grad
from ..common.alg_utils import *


def vanilla(env, env_maker, policy, baseline, n_iter=100, n_envs=mp.cpu_count(),
            n_batch=2000, last_iter=-1, gamma=0.99, gae_lambda=0.97,
            optimizer=None, scheduler=None, snapshot_saver=None):

    if optimizer is None:
        optimizer = torch.optim.Adam(policy.parameters())

    # Algorithm main loop
    with EnvPool(env, env_maker, n_envs=n_envs) as env_pool:
        for updt in trange(last_iter + 1, n_iter, desc="Training", unit="updt",
                           dynamic_ncols=True):
            logger.info("Starting iteration {}".format(updt))
            logger.logkv("Iteration", updt)
            
            logger.info("Start collecting samples")
            trajs = parallel_collect_experience(env_pool, policy, n_batch)
            
            logger.info("Computing policy gradient variables")
            all_obs, all_acts, all_advs, all_dists = compute_pg_vars(
                trajs, policy, baseline, gamma, gae_lambda
            )

            logger.info("Applying policy gradient")
            J0 = torch.mean(policy.dists(all_obs).log_prob(all_acts) * all_advs)

            if scheduler: scheduler.step(updt)
            optimizer.zero_grad()
            (-J0).backward()
            optimizer.step()
            
            logger.info("Updating baseline")
            baseline.update(trajs)

            logger.info("Logging information")
            logger.logkv("Objective", J0.item())
            log_reward_statistics(env)
            log_baseline_statistics(trajs)
            log_action_distribution_statistics(all_dists)
            logger.dumpkvs()

            if snapshot_saver is not None:
                logger.info("Saving snapshot")
                snapshot_saver.save_state(
                    updt+1,
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
                            scheduler=scheduler,
                            last_iter=updt,
                            gamma=gamma,
                            gae_lambda=gae_lambda
                        )
                    )
                )
