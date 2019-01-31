import torch
from torch.distributions.kl import kl_divergence as kl
from torch.utils.data import TensorDataset, DataLoader
from baselines import logger
from proj.utils.tqdm_util import trange
from proj.utils.saver import SnapshotSaver
from proj.common.models import ValueFunction
from proj.common.env_pool import EnvPool
from proj.common.sampling import parallel_collect_samples, compute_pg_vars
from proj.common.log_utils import save_config, log_reward_statistics, \
    log_val_fn_statistics, log_action_distribution_statistics


def ppo(env_maker, policy, val_fn=None,  steps=int(1e6), batch=2000,
        n_envs=16, gamma=0.99, gaelam=0.96, clip_ratio=0.2, pol_lr=3e-4,
        val_lr=1e-3, pol_iters=80, val_iters=80, target_kl=0.01,
        **saver_kwargs):

    if val_fn is None:
        val_fn = ValueFunction.from_policy(policy)

    save_config(locals())
    saver = SnapshotSaver(logger.get_dir(), locals(), **saver_kwargs)

    env = env_maker()
    policy = policy.pop('class')(env, **policy)
    val_fn = val_fn.pop('class')(env, **val_fn)
    pol_optim = torch.optim.Adam(policy.parameters(), lr=pol_lr)
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

            logger.info("Minimizing surrogate loss")
            with torch.no_grad():
                old_dists = policy(all_obs)
            old_logp = old_dists.log_prob(all_acts)
            dataset = TensorDataset(all_obs, all_acts, all_advs, old_logp)
            dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)
            for itr in range(pol_iters):
                for obs, acts, advs, logp in dataloader:
                    ratios = (policy(obs).log_prob(acts) - logp).exp()
                    min_advs = torch.where(
                        advs > 0,
                        (1+clip_ratio) * advs,
                        (1-clip_ratio) * advs
                    )
                    pol_optim.zero_grad()
                    (- torch.min(ratios*advs, min_advs)).mean().backward()
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
                val_optim.zero_grad()
                loss_fn(val_fn(all_obs), targets).backward()
                val_optim.step()

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
