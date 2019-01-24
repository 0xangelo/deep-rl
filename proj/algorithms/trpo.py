import torch, multiprocessing as mp
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from proj.utils import logger
from proj.utils.tqdm_util import trange
from proj.common.models import default_baseline
from proj.common.env_pool import EnvPool
from proj.common.sampling import parallel_collect_samples, compute_pg_vars
from proj.common.utils import conjugate_gradient, fisher_vec_prod, \
    flat_grad, line_search
from proj.common.log_utils import *


def trpo(env_maker, policy, baseline=None, steps=int(1e6), batch=2000,
         n_envs=mp.cpu_count(), gamma=0.99, gaelam=0.97, kl_frac=0.2,
         delta=0.01, val_iters=80, val_lr=1e-3, linesearch=True):

    if baseline is None:
        baseline = default_baseline(policy)

    logger.save_config(locals())
    env = env_maker.make()
    policy = policy.pop('class')(env, **policy)
    baseline = baseline.pop('class')(env, **baseline)
    val_optim = torch.optim.Adam(baseline.parameters(), lr=val_lr)
    loss_fn = torch.nn.MSELoss()

    # Algorithm main loop
    with EnvPool(env_maker, n_envs=n_envs) as env_pool:
        for updt in trange(steps // batch, desc="Training", unit="updt"):
            logger.info("Starting iteration {}".format(updt))
            logger.logkv("Iteration", updt)

            logger.info("Start collecting samples")
            buffer = parallel_collect_samples(env_pool, policy, batch)

            logger.info("Computing policy gradient variables")
            all_obs, all_acts, all_advs = compute_pg_vars(
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
            all_dists = policy(all_obs)
            all_logp = all_dists.log_prob(all_acts)
            old_dists = all_dists.detach()
            old_logp = old_dists.log_prob(all_acts)
            surr_loss = -((all_logp - old_logp).exp() * all_advs).mean()
            pol_grad = flat_grad(surr_loss, policy.parameters())

            logger.info("Computing truncated natural gradient")
            F_0 = lambda v: fisher_vec_prod(v, subsamp_obs, policy)
            descent_direction = conjugate_gradient(F_0, pol_grad)
            scale = torch.sqrt(
                2.0 * delta *
                (1. / (descent_direction.dot(F_0(descent_direction))) + 1e-8)
            )
            descent_step = descent_direction * scale

            if linesearch:
                logger.info("Performing line search")
                expected_improvement = pol_grad.dot(descent_step).item()

                def f_barrier(params):
                    vector_to_parameters(params, policy.parameters())
                    new_dists = policy(all_obs)
                    new_logp = new_dists.log_prob(all_acts)
                    surr_loss = -((new_logp - old_logp).exp() * all_advs).mean()
                    avg_kl = kl(old_dists, new_dists).mean().item()
                    return surr_loss.item() if avg_kl < delta else float('inf')

                new_params, expected_improvement, improvement = line_search(
                    f_barrier,
                    parameters_to_vector(policy.parameters()),
                    descent_step,
                    expected_improvement,
                    y0=surr_loss.item()
                )
                logger.logkv("ExpectedImprovement", expected_improvement)
                logger.logkv("ActualImprovement", improvement)
                logger.logkv("ImprovementRatio", improvement /
                             expected_improvement)
            else:
                new_params = parameters_to_vector(policy.parameters()) \
                             - descent_step
            vector_to_parameters(new_params, policy.parameters())

            logger.info("Updating baseline")
            targets = buffer["returns"]
            for _ in range(val_iters):
                val_optim.zero_grad()
                loss_fn(baseline(all_obs), targets).backward()
                val_optim.step()

            logger.info("Logging information")
            logger.logkv('TotalNSamples', (updt+1) * (batch - (batch % n_envs)))
            log_reward_statistics(env)
            log_baseline_statistics(buffer)
            log_action_distribution_statistics(old_dists)
            log_average_kl_divergence(old_dists, policy, all_obs)
            logger.dumpkvs()

            logger.info("Saving snapshot")
            logger.save_state(
                updt+1,
                dict(
                    alg=dict(last_iter=updt),
                    policy=policy.state_dict(),
                    baseline=baseline.state_dict(),
                    val_optim=val_optim.state_dict(),
                )
            )
