import torch
import numpy as np
from baselines import logger
from proj.utils.kfac import KFACOptimizer
from proj.utils.saver import SnapshotSaver
from proj.utils.tqdm_util import trange
from proj.common.models import WeightSharingAC, ValueFunction
from proj.common.utils import _NP_TO_PT, update_linear_schedule
from proj.common.log_utils import save_config, log_reward_statistics, \
    log_action_distribution_statistics, log_val_fn_statistics


def a2c_kfac(env_maker, policy, val_fn=None, total_samples=int(10e6), steps=20,
             n_envs=16, kfac={}, ent_coeff=0.01, vf_loss_coeff=0.5, gamma=0.99,
             log_interval=100, warm_start=None, **saver_kwargs):
    assert val_fn is None or not issubclass(policy['class'], WeightSharingAC), \
        "Choose between a weight sharing model or separate policy and val_fn"

    # handle default values
    kfac = {'eps': 1e-3, 'pi': True, 'alpha': 0.95, 'kl_clip': 1e-3, 'eta': 1.0,
            **kfac}
    if val_fn is None and not issubclass(policy['class'], WeightSharingAC):
        val_fn = ValueFunction.from_policy(policy)

    # save config and setup state saving
    save_config(locals())
    saver = SnapshotSaver(logger.get_dir(), locals(), **saver_kwargs)

    # initialize models and optimizer
    vec_env = env_maker(n_envs)
    policy = policy.pop('class')(vec_env, **policy)
    module_list = torch.nn.ModuleList(policy.modules())
    if val_fn is not None:
        val_fn = val_fn.pop('class')(vec_env, **val_fn)
        module_list.extend(val_fn.modules())
    optimizer = KFACOptimizer(module_list, **kfac)
    loss_fn = torch.nn.MSELoss()

    # load state if provided
    if warm_start is not None:
        if ':' in warm_start:
            warm_start, index = warm_start.split(':')
            _, state = SnapshotSaver(
                warm_start, latest_only=False).get_state(int(index))
        else:
            _, state = SnapshotSaver(warm_start).get_state()
        policy.load_state_dict(state['policy'])
        if 'optimizer' in state:
            optimizer.load_state_dict(state['optimizer'])
        del state

    # Algorith main loop
    if val_fn is None:
        compute_dists_vals = policy
    else:
        compute_dists_vals = lambda obs: policy(obs), val_fn(obs)

    ob_space, ac_space = vec_env.observation_space, vec_env.action_space
    obs = torch.from_numpy(vec_env.reset())
    with torch.no_grad():
        acts = policy.actions(obs)
    logger.info("Starting epoch {}".format(1))
    beg, end, stp = steps * n_envs, total_samples + steps*n_envs, steps * n_envs
    total_updates = total_samples // stp
    for samples in trange(beg, end, stp, desc="Training", unit="step"):
        all_obs = torch.empty((steps, n_envs) + ob_space.shape,
                              dtype=_NP_TO_PT[ob_space.dtype.type])
        all_acts = torch.empty((steps, n_envs) + ac_space.shape,
                               dtype=_NP_TO_PT[ac_space.dtype.type])
        all_rews = torch.empty((steps, n_envs))
        all_dones = torch.empty((steps, n_envs))

        with torch.no_grad():
            for i in range(steps):
                next_obs, rews, dones, _ = vec_env.step(acts.numpy())
                all_obs[i] = obs
                all_acts[i] = acts
                all_rews[i] = torch.from_numpy(rews)
                all_dones[i] = torch.from_numpy(dones.astype('f'))
                obs = torch.from_numpy(next_obs)

                acts = policy.actions(obs)

        all_obs = all_obs.reshape(stp, -1).squeeze()
        all_acts = all_acts.reshape(stp, -1).squeeze()

        # Sample Fisher curvature matrix
        with optimizer.record_stats():
            optimizer.zero_grad()
            all_dists, all_vals = compute_dists_vals(all_obs)
            logp = all_dists.log_prob(all_acts)
            noise = all_vals.detach() + 0.5*torch.randn_like(all_vals)
            (logp.mean() + loss_fn(all_vals, noise)).backward(retain_graph=True)
        del noise

        # Compute returns and advantages
        with torch.no_grad():
            _, next_vals = compute_dists_vals(obs)
        all_rets = all_rews.clone()
        all_rets[-1] += gamma * (1-all_dones[-1]) * next_vals
        for i in reversed(range(steps-1)):
            all_rets[i] += gamma * (1-all_dones[i]) * all_rets[i+1]
        all_rets = all_rets.flatten()
        all_advs = all_rets - all_vals.detach()

        # Compute loss
        updates = samples // stp
        ent_bonus = ent_coeff - (ent_coeff * ((updates-1)/float(total_updates)))
        pi_loss = - torch.mean(logp * all_advs)
        vf_loss = loss_fn(all_vals, all_rets)
        entropy = all_dists.entropy().mean()
        total_loss = pi_loss - ent_bonus*entropy + vf_loss_coeff*vf_loss

        update_linear_schedule(optimizer, updates-1, total_updates, 1.0)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if updates == 1 or updates % log_interval == 0:
            logger.logkv('Epoch', updates//log_interval + 1)
            logger.logkv('TotalNSamples', samples)
            log_reward_statistics(vec_env)
            log_val_fn_statistics(all_vals, all_rets)
            log_action_distribution_statistics(all_dists)
            logger.dumpkvs()
            logger.info("Starting epoch {}".format(updates//log_interval + 2))

        saver.save_state(
            updates,
            dict(
                alg=dict(last_updt=updates),
                policy=policy.state_dict(),
                val_fn=None if val_fn is None else val_fn.state_dict(),
                optimizer=optimizer.state_dict(),
            )
        )

    vec_env.close()
