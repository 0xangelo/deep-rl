import torch, multiprocessing as mp
from proj.utils import logger
from proj.utils.tqdm_util import trange
from proj.common.models import default_baseline
from proj.common.env_pool import EnvPool
from proj.common.sampling import parallel_collect_samples, compute_pg_vars
from proj.common.log_utils import *


def vanilla(env_maker, policy, baseline=None, steps=int(1e6), batch=2000,
            n_envs=mp.cpu_count(), gamma=0.99, gaelam=0.97,
            optimizer={}, val_iters=80, val_lr=1e-3):

    optimizer = {'class': torch.optim.Adam, **optimizer}
    if baseline is None:
        baseline = default_baseline(policy)

    logger.save_config(locals())

    env = env_maker()
    policy = policy.pop('class')(env, **policy)
    baseline = baseline.pop('class')(env, **baseline)
    pol_optim = optimizer.pop('class')(policy.parameters(), **optimizer)
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

            logger.info("Applying policy gradient")
            all_dists = policy(all_obs)
            old_dists = all_dists.detach()
            J0 = torch.mean(all_dists.log_prob(all_acts) * all_advs)
            pol_optim.zero_grad()
            (-J0).backward()
            pol_optim.step()

            logger.info("Updating baseline")
            targets = buffer["returns"]
            for _ in range(val_iters):
                val_optim.zero_grad()
                loss_fn(baseline(all_obs), targets).backward()
                val_optim.step()

            logger.info("Logging information")
            logger.logkv("Objective", J0.item())
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
