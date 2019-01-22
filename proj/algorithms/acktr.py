import torch, multiprocessing as mp
from proj.utils import logger
from proj.utils.kfac import KFACOptimizer
from proj.utils.tqdm_util import trange
from proj.common.models import default_baseline
from proj.common.env_pool import EnvPool
from proj.common.sampling import parallel_collect_samples, compute_pg_vars
from proj.common.log_utils import *
from proj.algorithms.trpo import line_search


DEFAULT_PIKFAC = dict(
    eps=1e-3, lr=1.0, pi=True, alpha=0.95, kl_clip=1e-3, eta=1.0
)
DEFAULT_VFKFAC = dict(
    eps=1e-3, lr=1.0, pi=True, alpha=0.95, kl_clip=0.01, eta=1.0
)

def acktr(env_maker, policy, baseline=None, steps=int(1e6), batch=2000,
          n_envs=mp.cpu_count(), gamma=0.99, gaelam=1.0, val_iters=10,
          pikfac={}, vfkfac={}):

    # handling default values
    pikfac = {**DEFAULT_PIKFAC, **pikfac}
    vfkfac = {**DEFAULT_VFKFAC, **vfkfac}
    if baseline is None:
        baseline = default_baseline(policy)

    logger.save_config(locals())

    env = env_maker.make()
    policy = policy.pop('class')(env, **policy)
    baseline = baseline.pop('class')(env, **baseline)
    pol_optim = KFACOptimizer(policy, **pikfac)
    # val_optim = KFACOptimizer(baseline, **vfkfac)
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
            with pol_optim.record_stats():
                policy.zero_grad()
                all_dists = policy.dists(all_obs)
                all_dists.log_prob(all_acts).mean().backward(retain_graph=True)

            policy.zero_grad()
            old_dists = all_dists.detach()
            surr_loss = -torch.mean(
                all_dists.likelihood_ratios(old_dists, all_acts) * all_advs
            )
            surr_loss.backward()
            pol_grad = [p.grad.clone() for p in policy.parameters()]
            pol_optim.step()
            expected_improvement = sum((
                (g * p.grad.data)).sum()
                 for g, p in zip(pol_grad, policy.parameters()
            )).item()

            kl_clip = pol_optim.kl_clip
            @torch.no_grad()
            def f_barrier(ratio):
                for p in policy.parameters():
                    p.data.add_(ratio, p.grad.data)
                new_dists = policy.dists(all_obs)
                surr_loss = -torch.mean(
                    new_dists.likelihood_ratios(old_dists, all_acts) * all_advs
                )
                avg_kl = kl(old_dists, new_dists).mean().item()
                return surr_loss.item() if avg_kl < kl_clip else float('inf')

            scale = line_search(
                f_barrier, 1, 1, expected_improvement, y0=surr_loss.item()
            )

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
            #     with val_optim.record_stats():
            #         baseline.zero_grad()
            #         loss_fn(baseline(all_obs), samples).backward()

            #     baseline.zero_grad()
            #     val_loss = loss_fn(baseline(all_obs), targets)
            #     val_loss.backward()
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
