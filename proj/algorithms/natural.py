import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.distributions.kl import kl_divergence as kl
from baselines import logger
from proj.utils.saver import SnapshotSaver
from proj.utils.tqdm_util import trange
from proj.utils.torch_util import flat_grad
from proj.common.models import ValueFunction
from proj.common.hf_util import conjugate_gradient, fisher_vec_prod, line_search
from proj.common.sampling import parallel_samples_collector, compute_pg_vars, \
    flatten_trajs
from proj.common.env_makers import VecEnvMaker
import proj.common.log_utils as logu


def natural(env, policy, val_fn=None, total_steps=int(1e6), steps=125,
            n_envs=16, gamma=0.99, gaelam=0.97, kl_frac=1.0, delta=0.01,
            val_iters=80, val_lr=1e-3, **saver_kwargs):

    if val_fn is None:
        val_fn = ValueFunction.from_policy(policy)
    logu.save_config(locals())
    saver = SnapshotSaver(logger.get_dir(), locals(), **saver_kwargs)

    vec_env = VecEnvMaker(env)(n_envs)
    policy = policy.pop('class')(vec_env, **policy)
    val_fn = val_fn.pop('class')(vec_env, **val_fn)
    val_optim = torch.optim.Adam(val_fn.parameters(), lr=val_lr)
    loss_fn = torch.nn.MSELoss()

    # Algorithm main loop
    collector = parallel_samples_collector(vec_env, policy, steps)
    beg, end, stp = steps * n_envs, total_steps + steps*n_envs, steps * n_envs
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

        # subsample for fisher vector product computation
        if kl_frac < 1.:
            n_samples = int(kl_frac*len(all_obs))
            indexes = torch.randperm(len(all_obs))[:n_samples]
            subsamp_obs = all_obs.index_select(0, indexes)
        else:
            subsamp_obs = all_obs

        logger.info("Computing policy gradient")
        all_dists = policy(all_obs)
        old_dists = all_dists.detach()
        pol_loss = torch.mean(all_dists.log_prob(all_acts) * all_advs).neg()
        pol_grad = flat_grad(pol_loss, policy.parameters())

        logger.info("Computing truncated natural gradient")
        F_0 = lambda v: fisher_vec_prod(v, subsamp_obs, policy)
        descent_direction = conjugate_gradient(F_0, pol_grad)
        scale = torch.sqrt(
            2.0 * delta *
            (1. / (descent_direction.dot(F_0(descent_direction))) + 1e-8)
        )
        descent_step = descent_direction * scale
        new_params = parameters_to_vector(policy.parameters()) - descent_step
        vector_to_parameters(new_params, policy.parameters())

        logger.info("Updating val_fn")
        for _ in range(val_iters):
            val_optim.zero_grad()
            loss_fn(val_fn(all_obs), all_rets).backward()
            val_optim.step()

        logger.info("Logging information")
        logger.logkv('TotalNSamples', samples)
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
                val_optim=val_optim.state_dict(),
            )
        )
        del all_obs, all_acts, all_advs, all_vals, all_rets, trajs

    vec_env.close()
