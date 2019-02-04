import torch
from torch.distributions.kl import kl_divergence as kl
from baselines import logger
from proj.utils.kfac import KFACOptimizer
from proj.utils.tqdm_util import trange
from proj.utils.saver import SnapshotSaver
from proj.common.models import ValueFunction
from proj.common.sampling import parallel_samples_collector, compute_pg_vars
from proj.common.utils import line_search
from proj.common.log_utils import save_config, log_reward_statistics, \
    log_val_fn_statistics, log_action_distribution_statistics, \
    log_average_kl_divergence


DEFAULT_PIKFAC = dict(eps=1e-3, pi=True, alpha=0.95, kl_clip=1e-2, eta=1.0)
DEFAULT_VFKFAC = dict(eps=1e-3, pi=True, alpha=0.95, kl_clip=1e-2, eta=1.0)

def acktr(env_maker, policy, val_fn=None, total_samples=int(1e6), steps=125,
          n_envs=16, gamma=0.99, gaelam=0.96, val_iters=20, pikfac={},
          vfkfac={}, warm_start=None, **saver_kwargs):

    # handling default values
    pikfac = {**DEFAULT_PIKFAC, **pikfac}
    vfkfac = {**DEFAULT_VFKFAC, **vfkfac}
    if val_fn is None:
        val_fn = ValueFunction.from_policy(policy)

    # save config and setup state saving
    save_config(locals())
    saver = SnapshotSaver(logger.get_dir(), locals(), **saver_kwargs)

    # initialize models and optimizer
    vec_env = env_maker(n_envs)
    policy = policy.pop('class')(vec_env, **policy)
    val_fn = val_fn.pop('class')(vec_env, **val_fn)
    pol_optim = KFACOptimizer(policy, **pikfac)
    val_optim = KFACOptimizer(val_fn, **vfkfac)
    loss_fn = torch.nn.MSELoss()

    # load state if provided
    if warm_start is not None:
        if ':' in warm_start:
            warm_start, index = warm_start.split(':')
            _, state = SnapshotSaver(
                warm_start, latest_only=False).get_state(int(index))
        else:
            _, state = SnapshotSaver(warm_start).get_state()
        policy.load_state_dict(state['policy'])
        val_fn.load_state_dict(state['val_fn'])
        if 'pol_optim' in state:
            pol_optim.load_state_dict(state['pol_optim'])
        if 'val_optim' in state:
            val_optim.load_state_dict(state['val_optim'])
        del state

    # Algorithm main loop
    collector = parallel_samples_collector(vec_env, policy, batch)
    beg, end, stp = steps * n_envs, total_samples + steps*n_envs, steps * n_envs
    for samples in trange(beg, end, stp, desc="Training", unit="step"):
        logger.info("Starting iteration {}".format(samples // stp))
        logger.logkv("Iteration", samples // stp)

        logger.info("Start collecting samples")
        trajs = next(collector)

        logger.info("Computing policy gradient variables")
        compute_pg_vars(trajs, policy, val_fn, gamma, gaelam)
        flatten_trajs(trajs, steps * n_envs)
        all_obs, all_acts, _, _, all_advs, all_vals, all_rets = trajs.values()
        all_obs, all_vals = all_obs[:-n_envs], all_vals[:-n_envs]

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
            f_barrier, 1, 1, expected_improvement, y0=surr_loss.item())
        logger.logkv("ExpectedImprovement", expected_improvement)
        logger.logkv("ActualImprovement", improvement)
        logger.logkv("ImprovementRatio", improvement / expected_improvement)
        for p in policy.parameters():
            p.data.add_(scale, p.grad.data)

        logger.info("Updating val_fn")
        for _ in range(val_iters):
            with val_optim.record_stats():
                val_fn.zero_grad()
                values = val_fn(all_obs)
                noise = values.detach() + 0.5*torch.randn_like(values)
                loss_fn(values, noise).backward(retain_graph=True)

            val_fn.zero_grad()
            val_loss = loss_fn(values, all_rets)
            val_loss.backward()
            val_optim.step()
        del values, noise

        logger.info("Logging information")
        logger.logkv('TotalNSamples', samples)
        log_reward_statistics2(vec_env)
        log_val_fn_statistics(all_vals, all_rets)
        log_action_distribution_statistics(old_dists)
        log_average_kl_divergence(old_dists, policy, all_obs)
        logger.dumpkvs()

        logger.info("Saving snapshot")
        saver.save_state(
            samples // stp,
            dict(
                alg=dict(last_iter=samples // stp),
                policy=policy.state_dict(),
                val_fn=val_fn.state_dict(),
                pol_optim=pol_optim.state_dict(),
                val_optim=val_optim.state_dict(),
            )
        )

    vec_env.close()
