import torch
from baselines import logger
from proj.utils.tqdm_util import trange
from proj.utils.saver import SnapshotSaver
from proj.common.models import ValueFunction
from proj.common.env_pool import EnvPool
from proj.common.sampling import parallel_collect_samples, compute_pg_vars
from proj.common.log_utils import save_config, log_reward_statistics, \
    log_val_fn_statistics, log_action_distribution_statistics, \
    log_average_kl_divergence


def vanilla(env_maker, policy, val_fn=None, steps=int(1e6), batch=2000,
            n_envs=16, gamma=0.99, gaelam=0.97, optimizer={}, val_iters=80,
            val_lr=1e-3, **saver_kwargs):

    optimizer = {'class': torch.optim.Adam, **optimizer}
    if val_fn is None:
        val_fn = ValueFunction.from_policy(policy)

    save_config(locals())
    saver = SnapshotSaver(logger.get_dir(), locals(), **saver_kwargs)

    env = env_maker()
    policy = policy.pop('class')(env, **policy)
    val_fn = val_fn.pop('class')(env, **val_fn)
    pol_optim = optimizer.pop('class')(policy.parameters(), **optimizer)
    val_optim = torch.optim.Adam(val_fn.parameters(), lr=val_lr)
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

            logger.info("Applying policy gradient")
            all_dists = policy(all_obs)
            old_dists = all_dists.detach()
            J0 = torch.mean(all_dists.log_prob(all_acts) * all_advs)
            pol_optim.zero_grad()
            (-J0).backward()
            pol_optim.step()

            logger.info("Updating val_fn")
            targets = buffer["returns"]
            for _ in range(val_iters):
                val_optim.zero_grad()
                loss_fn(val_fn(all_obs), targets).backward()
                val_optim.step()

            logger.info("Logging information")
            logger.logkv("Objective", J0.item())
            logger.logkv('TotalNSamples', (updt+1) * (batch - (batch % n_envs)))
            log_reward_statistics(env)
            log_val_fn_statistics(buffer["values"], buffer["returns"])
            log_action_distribution_statistics(old_dists)
            log_average_kl_divergence(old_dists, policy, all_obs)
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
