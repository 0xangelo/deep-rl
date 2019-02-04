import torch
from torch.distributions.kl import kl_divergence as kl
from torch.utils.data import TensorDataset, DataLoader
from baselines import logger
from proj.utils.tqdm_util import trange
from proj.utils.saver import SnapshotSaver
from proj.common.models import ValueFunction
from proj.common.sampling import parallel_samples_collector, compute_pg_vars
from proj.common.log_utils import save_config, log_reward_statistics, \
    log_val_fn_statistics, log_action_distribution_statistics


def ppo(env_maker, policy, val_fn=None, total_samples=int(1e6), steps=125,
        n_envs=16, gamma=0.99, gaelam=0.96, clip_ratio=0.2, pol_iters=80,
        val_iters=80, pol_lr=3e-4, val_lr=1e-3, target_kl=0.01, mb_size=32,
        **saver_kwargs):

    if val_fn is None:
        val_fn = ValueFunction.from_policy(policy)

    save_config(locals())
    saver = SnapshotSaver(logger.get_dir(), locals(), **saver_kwargs)

    vec_env = env_maker(n_envs)
    policy = policy.pop('class')(env, **policy)
    val_fn = val_fn.pop('class')(env, **val_fn)
    pol_optim = torch.optim.Adam(policy.parameters(), lr=pol_lr)
    val_optim = torch.optim.Adam(val_fn.parameters(), lr=val_lr)
    loss_fn = torch.nn.MSELoss()

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

        logger.info("Minimizing surrogate loss")
        with torch.no_grad():
            old_dists = policy(all_obs)
        old_logp = old_dists.log_prob(all_acts)
        min_advs = torch.where(
            all_advs > 0, (1+clip_ratio) * all_advs, (1-clip_ratio) * all_advs)
        dataset = TensorDataset(all_obs, all_acts, all_advs, min_advs, old_logp)
        dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)
        for itr in range(pol_iters):
            for obs, acts, advs, madv, logp in dataloader:
                ratios = (policy(obs).log_prob(acts) - logp).exp()
                pol_optim.zero_grad()
                (- torch.min(ratios*advs, madv)).mean().backward()
                pol_optim.step()

            with torch.no_grad():
                mean_kl = kl(old_dists, policy(all_obs)).mean().item()
            if mean_kl > 1.5 * target_kl:
                logger.info("Stopped at step {} due to reaching max kl".
                            format(itr+1))
                break
        logger.logkv("StopIter", itr+1)

        logger.info("Updating val_fn")
        for _ in range(val_iters):
            val_optim.zero_grad()
            loss_fn(val_fn(all_obs), all_rets).backward()
            val_optim.step()

        logger.info("Logging information")
        logger.logkv('TotalNSamples', samples)
        log_reward_statistics(vec_env)
        log_val_fn_statistics(all_vals, all_rets)
        log_action_distribution_statistics(old_dists)
        logger.logkv('MeanKL', mean_kl)
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
        del all_obs, all_acts, all_advs, all_vals, all_rets, trajs

    vec_env.close()
