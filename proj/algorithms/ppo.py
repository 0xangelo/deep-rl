import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions.kl import kl_divergence as kl
from baselines import logger
from proj.utils.saver import SnapshotSaver
from proj.utils.tqdm_util import trange
from proj.common.models import ValueFunction
from proj.common.sampling import (
    parallel_samples_collector,
    compute_pg_vars,
    flatten_trajs,
)
from proj.common.env_makers import VecEnvMaker
import proj.common.log_utils as logu


TOTAL_STEPS_DEFAULT = int(1e6)


def ppo(
    env,
    policy,
    val_fn=None,
    total_steps=TOTAL_STEPS_DEFAULT,
    steps=125,
    n_envs=16,
    gamma=0.99,
    gaelam=0.96,
    clip_ratio=0.2,
    pol_iters=80,
    val_iters=80,
    pol_lr=3e-4,
    val_lr=1e-3,
    target_kl=0.01,
    mb_size=100,
    **saver_kwargs
):
    val_fn = val_fn or ValueFunction.from_policy(policy)

    logu.save_config(locals())
    saver = SnapshotSaver(logger.get_dir(), locals(), **saver_kwargs)

    vec_env = VecEnvMaker(env)(n_envs)
    policy = policy.pop("class")(vec_env, **policy)
    val_fn = val_fn.pop("class")(vec_env, **val_fn)
    pol_optim = torch.optim.Adam(policy.parameters(), lr=pol_lr)
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

        logger.info("Minimizing surrogate loss")
        with torch.no_grad():
            old_dists = policy(all_obs)
        old_logp = old_dists.log_prob(all_acts)
        min_advs = torch.where(
            all_advs > 0, (1 + clip_ratio) * all_advs, (1 - clip_ratio) * all_advs
        )
        dataset = TensorDataset(all_obs, all_acts, all_advs, min_advs, old_logp)
        dataloader = DataLoader(dataset, batch_size=mb_size, shuffle=True)
        for itr in range(pol_iters):
            for obs, acts, advs, min_adv, logp in dataloader:
                ratios = (policy(obs).log_prob(acts) - logp).exp()
                pol_optim.zero_grad()
                (-torch.min(ratios * advs, min_adv)).mean().backward()
                pol_optim.step()

            with torch.no_grad():
                mean_kl = kl(old_dists, policy(all_obs)).mean().item()
            if mean_kl > 1.5 * target_kl:
                logger.info("Stopped at step {} due to reaching max kl".format(itr + 1))
                break
        logger.logkv("StopIter", itr + 1)

        logger.info("Updating val_fn")
        for _ in range(val_iters):
            val_optim.zero_grad()
            loss_fn(val_fn(all_obs), all_rets).backward()
            val_optim.step()

        logger.info("Logging information")
        logger.logkv("TotalNSamples", samples)
        logu.log_reward_statistics(vec_env)
        logu.log_val_fn_statistics(all_vals, all_rets)
        logu.log_action_distribution_statistics(old_dists)
        logger.logkv("MeanKL", mean_kl)
        logger.dumpkvs()

        logger.info("Saving snapshot")
        saver.save_state(
            index=samples // stp,
            state=dict(
                alg=dict(last_iter=samples // stp),
                policy=policy.state_dict(),
                val_fn=val_fn.state_dict(),
                pol_optim=pol_optim.state_dict(),
                val_optim=val_optim.state_dict(),
            ),
        )

    vec_env.close()
