import torch
import numpy as np
from baselines import logger
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from proj.utils.tqdm_util import trange
from proj.utils.saver import SnapshotSaver
from proj.common.models import ContinuousQFunction
from proj.common.log_utils import save_config, log_reward_statistics


class ReplayBuffer(object):
    def __init__(self, size, ob_space, ac_space):
        self.all_obs1 = torch.empty(size, *ob_space.shape)
        self.all_acts = torch.empty(size, *ac_space.shape)
        self.all_rews = torch.empty(size)
        self.all_obs2 = torch.empty(size, *ob_space.shape)
        self.all_dones = torch.empty(size)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, ob1, act, rew, ob2, done):
        self.all_obs1[self.ptr] = ob1
        self.all_acts[self.ptr] = act
        self.all_rews[self.ptr] = rew
        self.all_obs2[self.ptr] = ob2
        self.all_dones[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sampler(self, num_mbs, mb_size):
        dataset = TensorDataset(
            self.all_obs1[:self.size],
            self.all_acts[:self.size],
            self.all_rews[:self.size],
            self.all_obs2[:self.size],
            self.all_dones[:self.size]
        )
        sampler = RandomSampler(
            dataset, replacement=True, num_samples=num_mbs*mb_size)
        return DataLoader(dataset, batch_size=mb_size, sampler=sampler)


def td3(env_maker, policy, q_func=None, total_samples=int(5e5), gamma=0.99,
        steps=125, n_envs=8, replay_size=int(1e6), polyak=0.995, pi_lr=1e-3,
        qf_lr=1e-3, start_steps=10000, act_noise=0.1, mb_size=100,
        target_noise=0.2, noise_clip=0.5, policy_delay=2, epoch=5000,
        **saver_kwargs):

    if q_func is None:
        q_func = ContinuousQFunction.from_policy(policy)

    save_config(locals())
    saver = SnapshotSaver(logger.get_dir(), locals(), **saver_kwargs)

    vec_env = env_maker(n_envs)
    test_env = env_maker(train=False)
    ob_space, ac_space = vec_env.observation_space, vec_env.action_space
    pi_class = policy.pop('class')
    qf_class = q_func.pop('class')
    pi_targ = pi_class(vec_env, **policy)
    q1_targ = qf_class(vec_env, **q_func)
    q2_targ = qf_class(vec_env, **q_func)
    policy = pi_class(vec_env, **policy)
    q1func = qf_class(vec_env, **q_func)
    q2func = qf_class(vec_env, **q_func)
    replay = ReplayBuffer(replay_size, ob_space, ac_space)
    loss_fn = torch.nn.MSELoss()

    pi_optim = torch.optim.Adam(policy.parameters(), lr=pi_lr)
    q1_optim = torch.optim.Adam(q1func.parameters(), lr=qf_lr)
    q2_optim = torch.optim.Adam(q2func.parameters(), lr=qf_lr)

    for p, t in zip(policy.parameters(), pi_targ.parameters()):
        t.detach_().copy_(p)

    for q, t in zip(q1func.parameters(), q1_targ.parameters()):
        t.detach_().copy_(q)

    for q, t in zip(q2func.parameters(), q2_targ.parameters()):
        t.detach_().copy_(q)

    rand_uniform_actions = lambda _: np.stack(
        [ac_space.sample() for _ in range(n_envs)])

    act_low, act_high = map(torch.from_numpy, (ac_space.low, ac_space.high))
    @torch.no_grad()
    def noisy_policy_actions(obs):
        acts = policy(torch.from_numpy(obs))
        acts += act_noise*torch.randn_like(acts)
        return torch.max(torch.min(acts, act_high), act_low).numpy()

    obs1 = vec_env.reset()
    ob, don = test_env.reset(), False
    critic_updates = 0
    beg, end, stp = n_envs * steps, n_envs*steps + total_samples, n_envs * steps
    for samples in trange(beg, end, stp, desc="Training", unit="iter"):
        if samples <= start_steps:
            actions = rand_uniform_actions
        else:
            actions = noisy_policy_actions

        for _ in trange(steps, desc="Sampling", unit="step", leave=False):
            acts = actions(obs1)
            obs2, rews, dones, _ = vec_env.step(acts)
            as_tensors = map(
                torch.from_numpy, (obs1, acts, rews, obs2, dones.astype('f')))
            for ob1, act, rew, ob2, done in zip(*as_tensors):
                replay.store(ob1, act, rew, ob2, done)
            obs1 = obs2

        for ob_1, act_, rew_, ob_2, done_ in replay.sampler(stp, mb_size):
            with torch.no_grad():
                atarg = pi_targ(ob_2)
                atarg += torch.clamp(target_noise * torch.randn_like(atarg),
                                     -noise_clip, noise_clip)
                atarg = torch.max(torch.min(atarg, act_high), act_low)
                targs = rew_ + gamma*(1-done_) * torch.min(q1_targ(ob_2, atarg),
                                                           q2_targ(ob_2, atarg))
            q1_optim.zero_grad()
            loss_fn(q1func(ob_1, act_), targs).backward()
            q1_optim.step()

            q2_optim.zero_grad()
            loss_fn(q2func(ob_1, act_), targs).backward()
            q2_optim.step()

            critic_updates += 1
            if critic_updates % policy_delay == 0:
                pi_optim.zero_grad()
                (- q1func(ob_1, policy(ob_1))).mean().backward()
                pi_optim.step()

                for p, t in zip(policy.parameters(), pi_targ.parameters()):
                    t.data.mul_(polyak).add_(1 - polyak, p.data)

                for q, t in zip(q1func.parameters(), q1_targ.parameters()):
                    t.data.mul_(polyak).add_(1 - polyak, q.data)

                for q, t in zip(q2func.parameters(), q2_targ.parameters()):
                    t.data.mul_(polyak).add_(1 - polyak, q.data)

        if samples % epoch == 0:
            with torch.no_grad():
                for _ in range(10):
                    while not don:
                        act = policy(torch.from_numpy(ob)).numpy()
                        ob, _, don, _ = test_env.step(act)
                    don = False

            logger.logkv("Epoch", samples // epoch)
            logger.logkv("TotalNSamples", samples)
            log_reward_statistics(test_env)
            logger.dumpkvs()

        saver.save_state(
            samples // stp,
            dict(
                alg=dict(samples=samples),
                policy=policy.state_dict(),
                q1func=q1func.state_dict(),
                q2func=q2func.state_dict(),
                pi_optim=pi_optim.state_dict(),
                q1_optim=q1_optim.state_dict(),
                q2_optim=q2_optim.state_dict(),
                pi_targ=pi_targ.state_dict(),
                q1_targ=q1_targ.state_dict(),
                q2_targ=q2_targ.state_dict()
            )
        )
