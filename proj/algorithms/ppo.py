from proj.common.alg_utils import *
from proj.common.models import baseline_like_policy
from torch.utils.data import TensorDataset, DataLoader


def ppo(env_maker, policy, baseline=None, n_iter=100, n_envs=mp.cpu_count(),
            n_batch=2000, last_iter=-1, gamma=0.99, gaelam=0.97, clip_ratio=0.2,
            pol_lr=3e-4, val_lr=1e-3, pol_iters=80, val_iters=80, target_kl=0.01):

    logger.save_config(locals())

    env = env_maker.make()
    policy = policy.pop('class')(env, **policy)
    if baseline is None:
        baseline = baseline_like_policy(env, policy)
    else:
        baseline = baseline.pop('class')(env, **baseline)
    pol_optim = torch.optim.Adam(policy.parameters(), lr=pol_lr)
    val_optim = torch.optim.Adam(baseline.parameters(), lr=val_lr)

    if last_iter > -1:
        state = logger.get_state(last_iter+1)
        policy.load_state_dict(state['policy'])
        baseline.load_state_dict(state['baseline'])
        pol_optim.load_state_dict(state['pol_optim'])
        val_optim.load_state_dict(state['val_optim'])

    # Algorithm main loop
    with EnvPool(env_maker, n_envs=n_envs) as env_pool:
        for updt in trange(last_iter + 1, n_iter, desc="Training", unit="updt"):
            logger.info("Starting iteration {}".format(updt))
            logger.logkv("Iteration", updt)

            logger.info("Start collecting samples")
            buffer = parallel_collect_samples(env_pool, policy, n_batch)

            logger.info("Computing policy gradient variables")
            all_obs, all_acts, all_advs, old_dists = compute_pg_vars(
                buffer, policy, baseline, gamma, gaelam
            )

            logger.info("Minimizing surrogate loss")
            all_pars, extra = old_dists.params()
            dataset = TensorDataset(all_obs, all_acts, all_advs, all_pars)
            dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
            for itr in range(pol_iters):
                for obs, acts, advs, pars in dataloader:
                    piold = old_dists.fromparams(pars, extra)
                    pinew = policy.dists(obs)
                    ratio = pinew.likelihood_ratios(piold, acts)
                    min_advs = torch.where(
                        advs > 0,
                        (1 + clip_ratio) * advs,
                        (1 - clip_ratio) * advs
                    )
                    pol_optim.zero_grad()
                    torch.mean(- torch.min(ratio * advs, min_advs)).backward()
                    pol_optim.step()
                with torch.no_grad():
                    mean_kl = kl(old_dists, policy.dists(all_obs)).mean()
                if mean_kl > 1.5 * target_kl:
                    logger.info("Stopped at step {} due to reaching max kl".
                                format(itr))
                    break
            logger.logkv("StopIter", itr)

            logger.info("Updating baseline")
            loss_fn = torch.nn.MSELoss()
            for _ in range(val_iters):
                targets = 0.1 * buffer["baselines"] + 0.9 * buffer["returns"]
                val_optim.zero_grad()
                loss_fn(baseline(all_obs), targets).backward()
                val_optim.step()

            logger.info("Logging information")
            log_reward_statistics(env)
            log_baseline_statistics(buffer)
            log_action_distribution_statistics(buffer, policy)
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
