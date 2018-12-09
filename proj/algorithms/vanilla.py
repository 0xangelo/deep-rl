from ..common.alg_utils import *

def vanilla(env_maker, policy, baseline, n_iter=100, n_envs=mp.cpu_count(),
            n_batch=2000, last_iter=-1, gamma=0.99, gaelam=0.97,
            optimizer={'class': torch.optim.Adam}):

    env = env_maker.make()
    policy = policy.pop('class')(env, **policy)
    baseline = baseline.pop('class')(env, **baseline)
    optimizer = optimizer.pop('class')(policy.parameters(), **optimizer)

    if last_iter > -1:
        state = logger.get_state(last_iter+1)
        policy.load_state_dict(state['policy'])
        baseline.load_state_dict(state['baseline'])
        optimizer.load_state_dict(state['optimizer'])

    # Algorithm main loop
    with EnvPool(env_maker, n_envs=n_envs) as env_pool:
        for updt in trange(last_iter + 1, n_iter, desc="Training", unit="updt"):
            logger.info("Starting iteration {}".format(updt))
            logger.logkv("Iteration", updt)

            logger.info("Start collecting samples")
            buffer = parallel_collect_samples(env_pool, policy, n_batch)

            logger.info("Computing policy gradient variables")
            all_obs, all_acts, all_advs, all_dists = compute_pg_vars(
                buffer, policy, baseline, gamma, gaelam
            )

            logger.info("Applying policy gradient")
            J0 = torch.mean(policy.dists(all_obs).log_prob(all_acts) * all_advs)

            optimizer.zero_grad()
            (-J0).backward()
            optimizer.step()

            logger.info("Updating baseline")
            baseline.update(buffer)

            logger.info("Logging information")
            logger.logkv("Objective", J0.item())
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
                    optimizer=optimizer.state_dict(),
                )
            )
