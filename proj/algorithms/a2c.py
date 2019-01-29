import torch
import numpy as np
from baselines import logger
from proj.utils.tqdm_util import trange
from proj.utils.saver import SnapshotSaver
from proj.common.models import WeightSharingAC
from proj.common.env_pool import ShmEnvPool as EnvPool
from proj.common.sampling import samples_generator
from proj.common.log_utils import save_config, log_reward_statistics, \
    log_action_distribution_statistics, log_action_distribution_statistics


def a2c(env_maker, policy, vf=None, k=20, n_envs=16, gamma=0.99, optimizer={},
        max_grad_norm=1.0, ent_coeff=0.01, vf_loss_coeff=0.5,
        epoch_length=10000, samples=int(8e7), **saver_kwargs):
    assert vf is None or not isinstance(policy, WeightSharingAC), \
        "Choose between a weight sharing model or separate policy and baseline"

    optimizer = {'class': torch.optim.RMSprop, 'lr': 1e-3, **optimizer}

    save_config(locals())
    saver = SnapshotSaver(logger.get_dir(), locals(), **saver_kwargs)

    env = env_maker()
    policy = policy.pop('class')(env, **policy)
    param_list = torch.nn.ParameterList(policy.parameters())
    if vf is not None:
        vf = vf.pop('class')(env, **vf)
        params.extend(vf.parameters())
    optimizer = optimizer.pop('class')(param_list.parameters(), **optimizer)
    loss_fn = torch.nn.MSELoss()

    epoch = 0
    global_t = epoch * epoch_length
    # Algorith main loop
    with EnvPool(env_maker, n_envs=n_envs) as env_pool:
        if vf is None:
            compute_dists_vals = policy
        else:
            compute_dists_vals = lambda obs: policy(obs), vf(obs)
        gen = samples_generator(env_pool, policy, k, compute_dists_vals)

        for _ in trange(samples // epoch_length, desc="Training", unit="epch"):
            logger.info("Starting epoch {}".format(epoch))
            for t in trange(global_t, (epoch+1)*epoch_length, k*n_envs,
                            desc="Epoch", unit="updt", leave=False):
                all_acts, all_rews, all_dones, all_dists, all_vals, next_vals \
                    = next(gen)

                # Compute returns and advantages
                all_rets = all_rews.clone()
                all_rets[-1] += (1-all_dones[-1]) * next_vals
                for i in reversed(range(k-1)):
                    all_rets[i] += (1-all_dones[i]) * gamma * all_rets[i+1]
                all_advs = all_rets - all_vals

                # Compute loss
                log_li = all_dists.log_prob(all_acts.reshape(k*n_envs, -1))
                pi_loss = - torch.mean(log_li * all_advs.reshape(-1))
                vf_loss = loss_fn(all_vals.reshape(-1), all_rets.reshape(-1))
                entropy = all_dists.entropy().mean()
                total_loss = pi_loss - ent_coeff*entropy + vf_loss_coeff*vf_loss

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    param_list.parameters(), max_grad_norm)
                optimizer.step()

                logger.logkv_mean('|VfPred|', abs(all_vals.mean().item()))
                logger.logkv_mean('|VfTarget|', abs(all_rets.mean().item()))
                logger.logkv_mean('VfLoss', vf_loss.item())
                log_action_distribution_statistics(all_dists)

            global_t = t + k*n_envs
            epoch = global_t // epoch_length
            env_pool.flush()
            logger.logkv('Epoch', epoch)
            logger.logkv('TotalNSamples', global_t)
            log_reward_statistics(env)
            logger.dumpkvs()

            saver.save_state(
                epoch,
                dict(
                    alg=dict(last_epoch=epoch),
                    policy=policy.state_dict(),
                    vf=None if vf is None else vf.state_dict(),
                    optimizer=optimizer.state_dict(),
                )
            )
