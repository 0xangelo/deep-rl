import math
from proj.common.alg_utils import *
from proj.common.models import baseline_like_policy
from .kfac import KFAC


def acktr(env_maker, policy, baseline=None, n_iter=100, n_envs=mp.cpu_count(),
          n_batch=2000, last_iter=-1, gamma=0.99, gaelam=0.97, val_iters=10,
          pol_kfac={}, val_kfac={}):

    # handling default values
    pol_kfac = {
        **dict(eps=1e-3, alpha=0.95, constraint_norm=1e-3, pi=True, eta=1.0),
        **pol_kfac
    }
    val_kfac = {
        **dict(eps=1e-3, alpha=0.95, constraint_norm=0.01, pi=True, eta=1.0),
        **val_kfac
    }

    logger.save_config(locals())

    env = env_maker.make()
    policy = policy.pop('class')(env, **policy)
    if baseline is None:
        baseline = baseline_like_policy(env, policy)
    else:
        baseline = baseline.pop('class')(env, **baseline)
    pol_precd = KFAC(policy, **pol_kfac)
    pol_optim = torch.optim.SGD(policy.parameters(), lr=1)
    val_precd = KFAC(baseline, **val_kfac)
    val_optim = torch.optim.SGD(baseline.parameters(), lr=1)
    loss_fn = torch.nn.MSELoss()

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
            all_obs, all_acts, all_advs, all_dists = compute_pg_vars(
                buffer, policy, baseline, gamma, gaelam
            )

            logger.info("Updating policy using KFAC")
            with pol_precd.record_stats():
                policy.zero_grad()
                policy.dists(all_obs).log_prob(all_acts).mean().backward()

            pol_optim.zero_grad()
            J0 = torch.mean(policy.dists(all_obs).log_prob(all_acts) * all_advs)
            (-J0).backward()
            pol_precd.step()
            pol_optim.step()

            logger.info("Updating baseline")
            targets = buffer["returns"]
            for _ in range(val_iters):
                with torch.no_grad():
                    samples = baseline(all_obs) + torch.randn_like(all_advs)*0.5
                with val_precd.record_stats():
                    baseline.zero_grad()
                    loss_fn(baseline(all_obs), samples).backward()

                val_optim.zero_grad()
                loss_fn(baseline(all_obs), targets).backward()
                val_precd.step()
                val_optim.step()

            logger.info("Logging information")
            log_reward_statistics(env)
            with torch.no_grad():
                logger.logkv(
                    'ValueLoss', loss_fn(baseline(all_obs), targets).item()
                )
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

            del all_obs, all_acts, all_advs, all_dists, targets, buffer, J0, samples
