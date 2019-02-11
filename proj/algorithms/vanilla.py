import torch
from baselines import logger
from proj.utils.tqdm_util import trange
from proj.utils.saver import SnapshotSaver
from proj.common.models import ValueFunction
from proj.common.sampling import parallel_samples_collector, compute_pg_vars, \
    flatten_trajs
from proj.common.log_utils import save_config, log_reward_statistics, \
    log_val_fn_statistics, log_action_distribution_statistics, \
    log_average_kl_divergence


def vanilla(env_maker, policy, val_fn=None, total_samples=int(1e6), steps=125,
            n_envs=16, gamma=0.99, gaelam=0.97, optimizer={}, val_iters=80,
            val_lr=1e-3, **saver_kwargs):

    optimizer = {'class': torch.optim.Adam, **optimizer}
    if val_fn is None:
        val_fn = ValueFunction.from_policy(policy)

    save_config(locals())
    saver = SnapshotSaver(logger.get_dir(), locals(), **saver_kwargs)

    vec_env = env_maker(n_envs)
    policy = policy.pop('class')(vec_env, **policy)
    val_fn = val_fn.pop('class')(vec_env, **val_fn)
    pol_optim = optimizer.pop('class')(policy.parameters(), **optimizer)
    val_optim = torch.optim.Adam(val_fn.parameters(), lr=val_lr)
    loss_fn = torch.nn.MSELoss()

    # Algorithm main loop
    collector = parallel_samples_collector(vec_env, policy, steps)
    beg, end, stp = steps * n_envs, total_samples + steps*n_envs, steps * n_envs
    for samples in trange(beg, end, stp, desc="Training", unit="step"):
        logger.info("Starting iteration {}".format(samples // stp))
        logger.logkv("Iteration", samples // stp)

        logger.info("Start collecting samples")
        trajs = next(collector)

        logger.info("Computing policy gradient variables")
        compute_pg_vars(trajs, policy, val_fn, gamma, gaelam)
        flatten_trajs(trajs)
        all_obs, all_acts, _, _, all_advs, all_vals, all_rets = trajs.values()
        all_obs, all_vals = all_obs[:-n_envs], all_vals[:-n_envs]

        logger.info("Applying policy gradient")
        all_dists = policy(all_obs)
        old_dists = all_dists.detach()
        J0 = torch.mean(all_dists.log_prob(all_acts) * all_advs)
        pol_optim.zero_grad()
        (-J0).backward()
        pol_optim.step()

        logger.info("Updating val_fn")
        for _ in range(val_iters):
            val_optim.zero_grad()
            loss_fn(val_fn(all_obs), all_rets).backward()
            val_optim.step()

        logger.info("Logging information")
        logger.logkv("Objective", J0.item())
        logger.logkv('TotalNSamples', samples)
        log_reward_statistics(vec_env)
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
        del all_obs, all_acts, all_advs, all_vals, all_rets, trajs

    vec_env.close()
