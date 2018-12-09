from torch.nn.utils import parameters_to_vector, vector_to_parameters
from proj.common.utils import conjugate_gradient, fisher_vector_product, flat_grad
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


def trpo(env_maker, policy, baseline, n_iter=100, n_envs=mp.cpu_count(),
         n_batch=2000, last_iter=-1, gamma=0.99, gaelam=0.97,
         kl_frac=0.5, delta=0.01):

    env = env_maker.make()
    policy = policy.pop('class')(env, **policy)
    baseline = baseline.pop('class')(env, **baseline)

    if last_iter > -1:
        state = logger.get_state(last_iter+1)
        policy.load_state_dict(state['policy'])
        baseline.load_state_dict(state['baseline'])

    # Algorithm main loop
    with EnvPool(env_maker, n_envs=n_envs) as env_pool:
        for updt in trange(last_iter + 1, n_iter, desc="Training", unit="updt"):
            logger.info("Starting iteration {}".format(updt))
            logger.logkv("Iteration", updt)

            logger.info("Start collecting samples")
            buffer = parallel_collect_samples(env_pool, policy, n_batch)

            logger.info("Computing policy gradient variables")
            all_obs, all_acts, all_advs, all_dists = compute_pg_vars(
                buffer, policy, baseline, gamma, gaelam
            )

            # subsample for fisher vector product computation
            if kl_frac < 1.:
                n_samples = int(kl_frac*len(all_obs))
                indexes = torch.randperm(len(all_obs))[:n_samples]
                subsamp_obs = all_obs.index_select(0, indexes)
            else:
                subsamp_obs = all_obs

            logger.info("Computing policy gradient")
            new_dists = policy.dists(all_obs)
            surr_loss = -torch.mean(
                new_dists.likelihood_ratios(all_dists, all_acts) * all_advs
            )
            pol_grad = flat_grad(surr_loss, policy.parameters())

            logger.info("Computing truncated natural gradient")
            F_0 = lambda v: fisher_vector_product(v, subsamp_obs, policy)
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

            new_params = line_search(
                f_barrier,
                parameters_to_vector(policy.parameters()),
                descent_step,
                expected_improvement,
                y0=surr_loss.item()
            )
            vector_to_parameters(new_params, policy.parameters())

            logger.info("Updating baseline")
            baseline.update(buffer)

            logger.info("Logging information")
            log_reward_statistics(env)
            log_baseline_statistics(buffer)
            log_action_distribution_statistics(buffer, policy)
            logger.dumpkvs()

            logger.info("Saving snapshot")
            logger.save_state(
                updt+1,
                dict(
                    alg=dict(last_iter=updt),
                    policy=policy.state_dict(),
                    baseline=baseline.state_dict()
                )
            )
