"""
Implementation of 'Vanilla' Policy Gradient in PyTorch.
"""
import torch
from baselines import logger

from proj.utils.tqdm_util import trange
from proj.utils.saver import SnapshotSaver
from proj.common.models import ValueFunction
from proj.common.sampling import (
    parallel_samples_collector,
    compute_pg_vars,
    flatten_trajs,
)
from proj.common.env_makers import VecEnvMaker
import proj.common.log_utils as logu


TOTAL_STEPS_DEFAULT = int(1e6)


def vanilla(
    env,
    policy,
    val_fn=None,
    total_steps=TOTAL_STEPS_DEFAULT,
    steps=125,
    n_envs=16,
    gamma=0.99,
    gaelam=0.97,
    optimizer=None,
    val_iters=80,
    val_lr=1e-3,
    **saver_kwargs
):
    """
    Vanilla Policy Gradient

    env: instance of proj.common.env_makers.VecEnvMaker
    policy: instance of proj.common.models.Policy
    val_fn (optional): instance of proj.common.models.ValueFunction
    total_steps: total number of environment steps to take
    steps: number of steps to take in each environment per iteration
    n_envs: number of environment copies to run in parallel
    gamma: GAE discount parameter
    gaelam: GAE lambda exponential average parameter
    optimizer (optional): dictionary containing optimizer kwargs and/or class
    val_iters: number of optimization steps to update the critic per iteration
    val_lr: learning rate for critic optimizer
    saver_kwargs: keyword arguments for proj.utils.saver.SnapshotSaver
    """
    optimizer = optimizer or {}
    optimizer = {"class": torch.optim.Adam, **optimizer}
    val_fn = val_fn or ValueFunction.from_policy(policy)

    logu.save_config(locals())
    saver = SnapshotSaver(logger.get_dir(), locals(), **saver_kwargs)

    vec_env = VecEnvMaker(env)(n_envs)
    policy = policy.pop("class")(vec_env, **policy)
    val_fn = val_fn.pop("class")(vec_env, **val_fn)
    pol_optim = optimizer.pop("class")(policy.parameters(), **optimizer)
    val_optim = torch.optim.Adam(val_fn.parameters(), lr=val_lr)
    loss_fn = torch.nn.MSELoss()

    # Algorithm main loop
    collector = parallel_samples_collector(vec_env, policy, steps)
    beg, end, stp = steps * n_envs, total_steps + steps * n_envs, steps * n_envs
    for samples in trange(beg, end, stp, desc="Training", unit="step"):
        logger.info("Starting iteration {}".format(samples // stp))
        logger.logkv("Iteration", samples // stp)

        logger.info("Start collecting samples")
        trajs = next(collector)

        logger.info("Computing policy gradient variables")
        compute_pg_vars(trajs, val_fn, gamma, gaelam)
        flatten_trajs(trajs)
        all_obs, all_acts, _, _, all_advs, all_vals, all_rets = trajs.values()
        all_obs, all_vals = all_obs[:-n_envs], all_vals[:-n_envs]

        logger.info("Applying policy gradient")
        all_dists = policy(all_obs)
        old_dists = all_dists.detach()
        objective = torch.mean(all_dists.log_prob(all_acts) * all_advs)
        pol_optim.zero_grad()
        objective.neg().backward()
        pol_optim.step()

        logger.info("Updating val_fn")
        for _ in range(val_iters):
            val_optim.zero_grad()
            loss_fn(val_fn(all_obs), all_rets).backward()
            val_optim.step()

        logger.info("Logging information")
        logger.logkv("Objective", objective.item())
        logger.logkv("TotalNSamples", samples)
        logu.log_reward_statistics(vec_env)
        logu.log_val_fn_statistics(all_vals, all_rets)
        logu.log_action_distribution_statistics(old_dists)
        logu.log_average_kl_divergence(old_dists, policy, all_obs)
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
            ),
        )
        del all_obs, all_acts, all_advs, all_vals, all_rets, trajs

    vec_env.close()
