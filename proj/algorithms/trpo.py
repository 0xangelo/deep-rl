from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.distributions.kl import kl_divergence as kl
from torch.autograd import grad
from proj.common.utils import conjugate_gradient, fisher_vector_product
from proj.common.alg_utils import *


def line_search(f, x0, dx, expected_improvement, y0=None, accept_ratio=0.1,
                backtrack_ratio=0.8, max_backtracks=15, atol=1e-7):
    
    if y0 is None:
        y0 = f(x0)

    if expected_improvement >= atol:
        for exp in range(max_backtracks):
            ratio = backtrack_ratio ** exp
            x = x0 - ratio * dx
            y = f(x)
            actual_improvement = y0 - y
            # Armijo condition
            if actual_improvement >= expected_improvement * ratio * accept_ratio:
                logger.logkv("ExpectedImprovement", expected_improvement * ratio)
                logger.logkv("ActualImprovement", actual_improvement)
                logger.logkv("ImprovementRatio", actual_improvement /
                             (expected_improvement * ratio))
                return x

    logger.logkv("ExpectedImprovement", expected_improvement)
    logger.logkv("ActualImprovement", 0.)
    logger.logkv("ImprovementRatio", 0.)
    return x0


def trpo(env, env_maker, policy, baseline, n_iter=100, n_envs=mp.cpu_count(), 
         n_batch=2000, last_iter=-1, gamma=0.99, gae_lambda=0.97,
         kl_sample_frac=0.5, delta=0.01, snapshot_saver=None):

    # Algorithm main loop
    with EnvPool(env, env_maker, n_envs=n_envs) as env_pool:
        for updt in trange(last_iter + 1, n_iter, desc="Training",
                           unit="updt", file=std_out(), dynamic_ncols=True):
            logger.info("Starting iteration {}".format(updt))
            logger.logkv("Iteration", updt)
            
            logger.info("Start collecting samples")
            trajs = parallel_collect_samples(env_pool, policy, n_batch)
            
            logger.info("Computing policy gradient variables")
            all_obs, all_acts, all_advs, all_dists = compute_pg_vars(
                trajs, policy, baseline, gamma, gae_lambda
            )

            # subsample for kl divergence computation
            if kl_sample_frac < 1.:
                n_samples = int(kl_sample_frac*len(all_obs))
                indexes = torch.randperm(len(all_obs))[:n_samples]
                subsamp_obs = torch.index_select(all_obs, 0, indexes)
                subsamp_dists = policy.pdtype(
                    torch.index_select(all_dists.flatparam(), 0, indexes))
            else:
                subsamp_obs = all_obs
                subsamp_dists = all_dists

            logger.info("Computing policy gradient")
            new_dists = policy.dists(all_obs)
            surr_loss = -torch.mean(
                new_dists.likelihood_ratios(all_dists, all_acts) * all_advs
            )
            pol_grad = torch.cat([
                g.view(-1) for g in grad(surr_loss, policy.parameters())
            ])
            
            logger.info("Computing truncated natural gradient")
            avg_kl = lambda: kl(subsamp_dists, policy.dists(subsamp_obs)).mean()
            F_0 = lambda v: fisher_vector_product(v, avg_kl, policy)

            descent_direction = conjugate_gradient(F_0, pol_grad)
            scale = torch.sqrt(
                2.0 * delta *
                (1. / (descent_direction.dot(F_0(descent_direction))) + 1e-8)
            )
            descent_step = descent_direction * scale

            logger.info("Performing line search")
            expected_improvement = pol_grad.dot(descent_step).item()

            @torch.no_grad()
            def f_barrier(params):
                vector_to_parameters(params, policy.parameters())
                new_dists = policy.dists(all_obs)
                surr_loss = -torch.mean(
                    new_dists.likelihood_ratios(all_dists, all_acts) * all_advs)
                avg_kl = kl(all_dists, new_dists).mean().item()
                return surr_loss.item() + 1e100 * max(avg_kl - delta, 0.)
                
            flat_params = parameters_to_vector(policy.parameters())
            new_params = line_search(
                f_barrier,
                flat_params,
                descent_step,
                expected_improvement,
                y0=surr_loss.item()
            )
            vector_to_parameters(new_params, policy.parameters())

            logger.info("Updating baseline")
            baseline.update(trajs)

            logger.info("Logging information")
            with torch.no_grad():
                logger.logkv("MeanKL", avg_kl().item())
            log_reward_statistics(env)
            log_baseline_statistics(trajs)
            log_action_distribution_statistics(all_dists)
            logger.dumpkvs()

            if snapshot_saver is not None:
                logger.info("Saving snapshot")
                snapshot_saver.save_state(
                    updt,
                    dict(
                        alg=trpo,
                        alg_state=dict(
                            env_maker=env_maker,
                            policy=policy,
                            baseline=baseline,
                            n_iter=n_iter,
                            n_batch=n_batch,
                            n_envs=n_envs,
                            delta=delta,
                            kl_sample_frac=kl_sample_frac,
                            last_iter=updt,
                            gamma=gamma,
                            gae_lambda=gae_lambda
                        )
                    )
                )
