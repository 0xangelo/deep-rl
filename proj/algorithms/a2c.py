import torch
import numpy as np
from baselines import logger
from proj.utils.tqdm_util import trange
from proj.utils.saver import SnapshotSaver
from proj.common.models import WeightSharingAC, ValueFunction
from proj.common.env_pool import ShmEnvPool as EnvPool
from proj.common.sampling import samples_generator
from proj.common.log_utils import save_config, log_reward_statistics, \
    log_action_distribution_statistics, log_val_fn_statistics


def a2c(env_maker, policy, val_fn=None, n_envs=16, k=20, gamma=0.99,
        optimizer={}, max_grad_norm=0.5, ent_coeff=0.01, vf_loss_coeff=0.5,
        log_interval=100, samples=int(10e6), **saver_kwargs):
    assert val_fn is None or not issubclass(policy['class'], WeightSharingAC), \
        "Choose between a weight sharing model or separate policy and val_fn"

    optimizer = {'class': torch.optim.RMSprop, 'lr': 1e-3, 'eps': 1e-5,
                 'alpha': 0.99, **optimizer}
    if val_fn is None and not issubclass(policy['class'], WeightSharingAC):
        val_fn = ValueFunction.from_policy(policy)

    save_config(locals())
    saver = SnapshotSaver(logger.get_dir(), locals(), **saver_kwargs)

    env = env_maker()
    policy = policy.pop('class')(env, **policy)
    param_list = torch.nn.ParameterList(policy.parameters())
    if val_fn is not None:
        val_fn = val_fn.pop('class')(env, **val_fn)
        param_list.extend(val_fn.parameters())
    optimizer = optimizer.pop('class')(param_list.parameters(), **optimizer)
    loss_fn = torch.nn.MSELoss()

    batch = n_envs * k
    # Algorith main loop
    with EnvPool(env_maker, n_envs=n_envs) as env_pool:
        if val_fn is None:
            compute_dists_vals = policy
        else:
            compute_dists_vals = lambda obs: policy(obs), val_fn(obs)
        gen = samples_generator(env_pool, policy, k, compute_dists_vals)

        logger.info("Starting epoch {}".format(1))
        for updt in trange(1, samples//batch + 1, desc="Training", unit="updt"):
            all_acts, all_rews, all_dones, all_dists, all_vals, next_vals \
                = next(gen)

            # Compute returns and advantages
            all_rets = all_rews.clone()
            all_rets[-1] += gamma * (1-all_dones[-1]) * next_vals
            for i in reversed(range(k-1)):
                all_rets[i] += gamma * (1-all_dones[i]) * all_rets[i+1]
            all_advs = all_rets - all_vals.detach()

            # Compute loss
            log_li = all_dists.log_prob(all_acts.reshape(batch, -1).squeeze())
            pi_loss = - torch.mean(log_li * all_advs.flatten())
            vf_loss = loss_fn(all_vals.flatten(), all_rets.flatten())
            entropy = all_dists.entropy().mean()
            total_loss = pi_loss - ent_coeff*entropy + vf_loss_coeff*vf_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                param_list.parameters(), max_grad_norm)
            optimizer.step()

            if updt == 1 or updt % log_interval == 0:
                env_pool.flush()
                logger.logkv('Epoch', updt//log_interval + 1)
                logger.logkv('TotalNSamples', updt * batch)
                log_reward_statistics(env)
                log_val_fn_statistics(all_vals.flatten(), all_rets.flatten())
                log_action_distribution_statistics(all_dists)
                logger.dumpkvs()
                logger.info("Starting epoch {}".format(updt//log_interval + 2))

            saver.save_state(
                updt,
                dict(
                    alg=dict(last_updt=updt),
                    policy=policy.state_dict(),
                    val_fn=None if val_fn is None else val_fn.state_dict(),
                    optimizer=optimizer.state_dict(),
                )
            )
