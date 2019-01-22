import torch, multiprocessing as mp
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from proj.utils import logger
from proj.utils.kfac import KFAC
from proj.utils.tqdm_util import trange
from proj.common.models import default_baseline
from proj.common.env_pool import EnvPool
from proj.common.sampling import parallel_collect_samples, compute_pg_vars
from proj.common.log_utils import *
from proj.algorithms.trpo import line_search


def acktr(env_maker, policy, baseline=None, steps=int(1e6), batch=2000,
          n_envs=mp.cpu_count(), gamma=0.99, gaelam=1.0, val_iters=10,
          pol_kfac={}, val_kfac={}):

    # handling default values
    pol_kfac = {
        **dict(eps=1e-3, alpha=0.95, kl_clip=1e-3, pi=True, eta=1.0),
        **pol_kfac
    }
    val_kfac = {
        **dict(eps=1e-3, alpha=0.95, kl_clip=0.01, pi=True, eta=1.0),
        **val_kfac
    }
    if baseline is None:
        baseline = default_baseline(policy)

    logger.save_config(locals())

    env = env_maker.make()
    policy = policy.pop('class')(env, **policy)
    baseline = baseline.pop('class')(env, **baseline)
    pol_precd = KFAC(policy, **pol_kfac)
    pol_optim = torch.optim.SGD(policy.parameters(), lr=1)
    # val_precd = KFAC(baseline, **val_kfac)
    # val_optim = torch.optim.SGD(baseline.parameters(), lr=1)
    val_optim = torch.optim.Adam(baseline.parameters())
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

            logger.info("Updating policy using KFAC")
            with pol_precd.record_stats():
                all_dists = policy.dists(all_obs)
                policy.zero_grad()
                all_dists.log_prob(all_acts).mean().backward(retain_graph=True)

            pol_optim.zero_grad()
            old_dists = all_dists.detach()
            surr_loss = -torch.mean(
                all_dists.likelihood_ratios(old_dists, all_acts) * all_advs
            )
            surr_loss.backward()
            pol_grad = torch.cat(
                [p.grad.detach().reshape((-1,)) for p in policy.parameters()]
            )
            pol_precd.step()
            descent_step = torch.cat(
                [p.grad.detach().reshape((-1,)) for p in policy.parameters()]
            )
            expected_improvement = pol_grad.dot(descent_step).item()

            kl_clip = pol_precd.kl_clip
            @torch.no_grad()
            def f_barrier(params):
                vector_to_parameters(params, policy.parameters())
                new_dists = policy.dists(all_obs)
                surr_loss = -torch.mean(
                    new_dists.likelihood_ratios(old_dists, all_acts) * all_advs
                )
                avg_kl = kl(old_dists, new_dists).mean().item()
                return surr_loss.item() if avg_kl < kl_clip else float('inf')

            new_params = line_search(
                f_barrier,
                parameters_to_vector(policy.parameters()),
                descent_step,
                expected_improvement,
                y0=surr_loss.item()
            )
            vector_to_parameters(new_params, policy.parameters())

            logger.info("Updating baseline")
            targets = buffer["returns"]
            for _ in range(80):
                val_optim.zero_grad()
                val_loss = loss_fn(baseline(all_obs), targets)
                val_loss.backward()
                val_optim.step()

            # targets = buffer["returns"]
            # for _ in range(val_iters):
            #     with torch.no_grad():
            #         samples = baseline(all_obs) + torch.randn_like(all_advs)*0.5
            #     with val_precd.record_stats():
            #         baseline.zero_grad()
            #         loss_fn(baseline(all_obs), samples).backward()

            #     val_optim.zero_grad()
            #     val_loss = loss_fn(baseline(all_obs), targets)
            #     val_loss.backward()
            #     val_precd.step()
            #     val_optim.step()

            logger.info("Logging information")
            logger.logkv('ValueLoss', val_loss.item())
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
                    pol_optim=pol_optim.state_dict(),
                    val_optim=val_optim.state_dict(),
                )
            )

            del all_obs, all_acts, all_advs, all_dists, targets, buffer# , samples
