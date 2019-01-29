import torch
from torch.distributions.kl import kl_divergence as kl
from baselines import logger
from proj.utils.kfac import KFACOptimizer
from proj.utils.tqdm_util import trange
from proj.utils.saver import SnapshotSaver
from proj.common.models import default_baseline
from proj.common.env_pool import EnvPool
from proj.common.sampling import parallel_collect_samples, compute_pg_vars
from proj.common.utils import line_search
from proj.common.log_utils import save_config, log_reward_statistics, \
    log_baseline_statistics, log_action_distribution_statistics, \
    log_average_kl_divergence


DEFAULT_PIKFAC = dict(eps=1e-3, pi=True, alpha=0.95, kl_clip=1e-2, eta=1.0)
DEFAULT_VFKFAC = dict(eps=1e-3, pi=True, alpha=0.95, kl_clip=1e-2, eta=1.0)

def acktr(env_maker, policy, baseline=None, steps=int(1e6), batch=4000,
          n_envs=16, gamma=0.99, gaelam=0.96, val_iters=20, pikfac={},
          vfkfac={}, **saver_kwargs):

    # handling default values
    pikfac = {**DEFAULT_PIKFAC, **pikfac}
    vfkfac = {**DEFAULT_VFKFAC, **vfkfac}
    if baseline is None:
        baseline = default_baseline(policy)

    save_config(locals())
    saver = SnapshotSaver(logger.get_dir(), locals(), **saver_kwargs)

    env = env_maker()
    policy = policy.pop('class')(env, **policy)
    baseline = baseline.pop('class')(env, **baseline)
    pol_optim = KFACOptimizer(policy, **pikfac)
    val_optim = KFACOptimizer(baseline, **vfkfac)
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

            logger.info("Computing natural gradient using KFAC")
            with pol_optim.record_stats():
                policy.zero_grad()
                all_dists = policy(all_obs)
                all_logp = all_dists.log_prob(all_acts)
                all_logp.mean().backward(retain_graph=True)

            policy.zero_grad()
            old_dists, old_logp = all_dists.detach(), all_logp.detach()
            surr_loss = -((all_logp - old_logp).exp() * all_advs).mean()
            surr_loss.backward()
            pol_grad = [p.grad.clone() for p in policy.parameters()]
            pol_optim.step()
            expected_improvement = sum((
                (g * p.grad.data).sum()
                for g, p in zip(pol_grad, policy.parameters())
            )).item()
            del pol_grad, all_dists, all_logp

            logger.info("Performing line search")
            kl_clip = pol_optim.kl_clip
            def f_barrier(scale):
                for p in policy.parameters():
                    p.data.add_(scale, p.grad.data)
                new_dists = policy(all_obs)
                for p in policy.parameters():
                    p.data.sub_(scale, p.grad.data)
                new_logp = new_dists.log_prob(all_acts)
                surr_loss = -((new_logp - old_logp).exp() * all_advs).mean()
                avg_kl = kl(old_dists, new_dists).mean().item()
                return surr_loss.item() if avg_kl < kl_clip else float('inf')

            scale, expected_improvement, improvement = line_search(
                f_barrier, 1, 1, expected_improvement, y0=surr_loss.item()
            )
            logger.logkv("ExpectedImprovement", expected_improvement)
            logger.logkv("ActualImprovement", improvement)
            logger.logkv("ImprovementRatio", improvement / expected_improvement)
            for p in policy.parameters():
                p.data.add_(scale, p.grad.data)

            logger.info("Updating baseline")
            targets = buffer["returns"]
            for _ in range(val_iters):
                with val_optim.record_stats():
                    baseline.zero_grad()
                    values = baseline(all_obs)
                    noise = values.detach() + torch.randn_like(values) * 0.5
                    loss_fn(values, noise).backward(retain_graph=True)

                baseline.zero_grad()
                val_loss = loss_fn(values, targets)
                val_loss.backward()
                val_optim.step()
            del values, noise

            logger.info("Logging information")
            logger.logkv('TotalNSamples', (updt+1) * (batch - (batch % n_envs)))
            log_reward_statistics(env)
            log_baseline_statistics(buffer)
            log_action_distribution_statistics(old_dists)
            log_average_kl_divergence(old_dists, policy, all_obs)
            logger.dumpkvs()

            logger.info("Saving snapshot")
            saver.save_state(
                updt+1,
                dict(
                    alg=dict(last_iter=updt),
                    policy=policy.state_dict(),
                    baseline=baseline.state_dict(),
                    pol_optim=pol_optim.state_dict(),
                    val_optim=val_optim.state_dict(),
                )
            )
