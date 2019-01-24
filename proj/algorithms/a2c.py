import torch, numpy as np, multiprocessing as mp
from torch.optim import RMSprop
from torch.nn.utils import clip_grad_norm_
from collections import defaultdict
from proj.utils import logger
from proj.utils.tqdm_util import trange
from proj.common.env_pool import EnvPool
from proj.common.sampling import samples_generator
from proj.common.log_utils import *
from proj.common.models import WeightSharingAC


def a2c(env_maker, policy, vf=None, k=20, n_envs=mp.cpu_count(),
        gamma=0.99, optimizer={}, max_grad_norm=1.0, ent_coeff=0.01,
        vf_loss_coeff=0.5, epoch_length=10000, samples=int(8e7)):
    assert vf is None or not isinstance(policy, WeightSharingAC), \
        "Choose between a weight sharing model or separate policy and baseline"

    optimizer = {'class': RMSprop, 'lr': 1e-3, **optimizer}

    logger.save_config(locals())

    env = env_maker.make()
    policy = policy.pop('class')(env, **policy)
    param_list = torch.nn.ParameterList(policy.parameters())
    if vf is not None:
        vf = vf.pop('class')(env, **vf)
        params.extend(vf.parameters())
    optimizer = optimizer.pop('class')(param_list.parameters(), **optimizer)
    loss_fn = torch.nn.MSELoss()

    epoch = 0
    global_t = epoch * epoch_length
    loggings = defaultdict(list)

    # Algorith main loop
    with EnvPool(env_maker, n_envs=n_envs) as env_pool:
        if vf is None:
            compute_dists_vals = policy
        else:
            compute_dists_vals = lambda obs: policy(obs), vf(obs)
        gen = samples_generator(env_pool, policy, k, compute_dists_vals)

        logger.info("Starting epoch {}".format(epoch))
        for _ in trange(samples // epoch_length, desc="Training", unit="updt"):
            for t in trange(global_t, (epoch+1)*epoch_length, k*n_envs,
                            desc="Epoch", unit="iter", leave=False):
                all_acts, all_rews, all_dones, all_dists, all_vals, next_vals \
                    = next(gen)

                # Compute returns and advantages
                all_rets = torch.empty_like(all_rews)
                all_rets[-1] = (1 - all_dones[-1]) * next_vals
                for i in reversed(range(k-1)):
                    all_rets[i] = all_rews[i] + \
                                  (1 - all_dones[i]) * gamma * all_rets[i+1]
                all_advs = all_rets - all_vals

                # Compute loss
                log_li = all_dists.log_prob(all_acts.reshape(k*n_envs, -1))
                pi_loss = -torch.mean(log_li * all_advs.reshape(-1))
                vf_loss = loss_fn(all_vals.reshape(-1), all_rets.reshape(-1))
                entropy = all_dists.entropy().mean()
                total_loss = pi_loss - ent_coeff*entropy + vf_loss_coeff*vf_loss

                optimizer.zero_grad()
                total_loss.backward()
                clip_grad_norm_(param_list.parameters(), max_grad_norm)
                optimizer.step()

                loggings["vf_loss"].append(vf_loss.detach().numpy())
                loggings["vf_preds"].append(all_vals.detach().numpy())
                loggings["vf_targs"].append(all_rets.numpy())
                loggings["dists"].append(all_dists.flat_params.detach())

            global_t = t + k*n_envs
            epoch = global_t // epoch_length
            env_pool.flush()
            logger.logkv('Epoch', epoch)
            logger.logkv('TotalNSamples', global_t)
            logger.logkv('|VfPred|', np.mean(np.abs(loggings["vf_preds"])))
            logger.logkv('|VfTarget|', np.mean(np.abs(loggings["vf_targs"])))
            logger.logkv('VfLoss', np.mean(loggings["vf_loss"]))
            log_reward_statistics(env)
            log_action_distribution_statistics(
                policy.pdtype.from_flat(torch.cat(loggings["dists"]))
            )
            logger.dumpkvs()

            logger.save_state(
                epoch,
                dict(
                    alg=dict(last_epoch=epoch),
                    policy=policy.state_dict(),
                    vf=None if vf is None else vf.state_dict(),
                    optimizer=optimizer.state_dict(),
                )
            )

            loggings = defaultdict(list)
            logger.info("Starting epoch {}".format(epoch))
