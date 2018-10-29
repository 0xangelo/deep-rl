from torch.distributions.kl import kl_divergence as kl
from torch.nn.utils import parameters_to_vector as theta
from ..common.utils import conjugate_gradient, fisher_vector_product, flat_grad
from ..common.alg_utils import *


def natural(env, env_maker, policy, baseline, n_iter=100, n_envs=mp.cpu_count(),
            n_batch=2000, last_iter=-1, gamma=0.99, gae_lambda=0.97,
            kl_frac=0.4, optimizer=None, scheduler=None, snapshot_saver=None):

    if optimizer is None:
        optimizer = torch.optim.Adam(policy.parameters())
        scheduler = torch.optimlr_scheduler.ExponentialLR(optimizer, 1)

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

            # subsample for kl divergence computation
            if kl_frac < 1.:
                n_samples = int(kl_frac*len(all_obs))
                indexes = torch.randperm(len(all_obs))[:n_samples]
                subsamp_obs = all_obs.index_select(0, indexes)
                subsamp_dists = policy.pdtype(
                    all_dists.flatparam().index_select(0, indexes))
            else:
                subsamp_obs = all_obs
                subsamp_dists = all_dists

            logger.info("Computing policy gradient")
            J0 = torch.mean(policy.dists(all_obs).log_prob(all_acts) * all_advs)
            pol_grad = flat_grad(J0, policy.parameters())

            logger.info("Applying truncated natural gradient")
            F_0 = lambda v: fisher_vector_product(v, subsamp_obs, policy)
            natural_gradient = conjugate_gradient(F_0, pol_grad)

            scheduler.step(updt)
            optimizer.zero_grad()
            theta(policy.parameters()).matmul(-natural_gradient).backward()
            optimizer.step()

            logger.info("Updating baseline")
            baseline.update(trajs)

            logger.info("Logging information")
            with torch.no_grad():
                avg_kl = kl(subsamp_dists, policy.dists(subsamp_obs)).mean()
                logger.logkv("MeanKL", avg_kl.item())
            log_reward_statistics(env)
            log_baseline_statistics(trajs)
            log_action_distribution_statistics(all_dists)
            logger.dumpkvs()

            if snapshot_saver is not None:
                logger.info("Saving snapshot")
                snapshot_saver.save_state(
                    updt+1,
                    dict(
                        alg=dict(last_iter=updt),
                        policy=policy.state_dict(),
                        baseline=baseline.state_dict(),
                        optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(),
                    )
                )
