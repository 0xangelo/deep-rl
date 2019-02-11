import torch
from torch.distributions.kl import kl_divergence as kl
from baselines import logger
from proj.utils.kfac import KFACOptimizer
from proj.utils.saver import SnapshotSaver
from proj.utils.tqdm_util import trange
from proj.common.models import ValueFunction
from proj.common.env_pool import EnvPool
from proj.common.sampling import parallel_collect_samples, compute_pg_vars
from proj.common.log_utils import save_config, log_reward_statistics, \
    log_val_fn_statistics, log_action_distribution_statistics


DEFAULT_PIKFAC = dict(eps=1e-3, pi=True, alpha=0.95, kl_clip=3e-4, eta=1.0)
DEFAULT_VFKFAC = dict(eps=1e-3, pi=True, alpha=0.95, kl_clip=7e-3, eta=1.0)

def ppo_kfac(env_maker, policy, val_fn=None,  steps=int(1e6), batch=2000,
             n_envs=16, gamma=0.99, gaelam=0.96, clip_ratio=0.2, pikfac={},
              vfkfac={}, pol_iters=10, val_iters=20, target_kl=0.01,
             **saver_kwargs):

    pikfac = {**DEFAULT_PIKFAC, **pikfac}
    vfkfac = {**DEFAULT_VFKFAC, **vfkfac}
    if val_fn is None:
        val_fn = ValueFunction.from_policy(policy)

    save_config(locals())
    saver = SnapshotSaver(logger.get_dir(), locals(), **saver_kwargs)

    env = env_maker()
    policy = policy.pop('class')(env, **policy)
    val_fn = val_fn.pop('class')(env, **val_fn)
    pol_optim = KFACOptimizer(policy, **pikfac)
    val_optim = KFACOptimizer(val_fn, **vfkfac)
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
                buffer, policy, val_fn, gamma, gaelam
            )

            logger.info("Minimizing surrogate loss")
            with torch.no_grad():
                old_dists = policy(all_obs)
            old_logp = old_dists.log_prob(all_acts)
            min_advs = torch.where(all_advs > 0,
                                   (1+clip_ratio) * all_advs,
                                   (1-clip_ratio) * all_advs)
            for itr in range(pol_iters):
                with pol_optim.record_stats():
                    policy.zero_grad()
                    new_logp = policy(all_obs).log_prob(all_acts)
                    new_logp.mean().backward(retain_graph=True)

                ratios = (new_logp - old_logp).exp()
                policy.zero_grad()
                (- torch.min(ratios*all_advs, min_advs)).mean().backward()
                pol_optim.step()

                with torch.no_grad():
                    mean_kl = kl(old_dists, policy(all_obs)).mean().item()
                if mean_kl > 1.5 * target_kl:
                    logger.info("Stopped at step {} due to reaching max kl".
                                format(itr+1))
                    break
            logger.logkv("StopIter", itr+1)

            logger.info("Updating val_fn")
            targets = buffer["returns"]
            for _ in range(val_iters):
                with val_optim.record_stats():
                    val_fn.zero_grad()
                    values = val_fn(all_obs)
                    noise = values.detach() + torch.randn_like(values) * 0.5
                    loss_fn(values, noise).backward(retain_graph=True)

                val_fn.zero_grad()
                val_loss = loss_fn(values, targets)
                val_loss.backward()
                val_optim.step()
            del values, noise

            logger.info("Logging information")
            logger.logkv('TotalNSamples', (updt+1) * (batch - (batch % n_envs)))
            log_reward_statistics(env)
            log_val_fn_statistics(buffer["values"], buffer["returns"])
            log_action_distribution_statistics(old_dists)
            logger.logkv('MeanKL', mean_kl)
            logger.dumpkvs()

            logger.info("Saving snapshot")
            saver.save_state(
                updt+1,
                dict(
                    alg=dict(last_iter=updt),
                    policy=policy.state_dict(),
                    val_fn=val_fn.state_dict(),
                    pol_optim=pol_optim.state_dict(),
                    val_optim=val_optim.state_dict(),
                )
            )
