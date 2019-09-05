import torch
import numpy as np
from baselines import logger
from proj.utils.saver import SnapshotSaver
from proj.utils.tqdm_util import trange
from proj.utils.torch_util import update_polyak
from proj.common.models import ContinuousQFunction
from proj.common.sampling import ReplayBuffer
from proj.common.log_utils import save_config, log_reward_statistics
from proj.common.env_makers import VecEnvMaker


def load_state_dicts(namespace, state):
    from inspect import ismethod
    for key, val in state.items():
        if key in namespace:
            if ismethod(getattr(namespace[key], 'load_state_dict', None)):
                namespace[key].load_state_dict(val)
            elif isinstance(val, torch.Tensor):
                namespace[key].data.copy_(val)


def sac2(env, policy, q_func=None, total_samples=int(5e5), gamma=0.99,
         replay_size=int(1e6), polyak=0.995, start_steps=10000, epoch=5000,
         mb_size=100, lr=1e-3, alpha=0.2, target_entropy=None, reward_scale=1.0,
         updates_per_step=1.0, load_from=None, **saver_kwargs):

    # Set and save experiment hyperparameters
    if q_func is None:
        q_func = ContinuousQFunction.from_policy(policy)
    save_config(locals())
    saver = SnapshotSaver(logger.get_dir(), locals(), **saver_kwargs)

    # Initialize environments, models, replay buffer and entropy coefficient
    vec_env = VecEnvMaker(env)()
    test_env = VecEnvMaker(env)(train=False)
    ob_space, ac_space = vec_env.observation_space, vec_env.action_space
    pi_class, pi_args = policy.pop('class'), policy
    qf_class, qf_args = q_func.pop('class'), q_func
    policy = pi_class(vec_env, **pi_args)
    q1func = qf_class(vec_env, **qf_args)
    q2func = qf_class(vec_env, **qf_args)
    replay = ReplayBuffer(replay_size, ob_space, ac_space)
    if target_entropy is not None:
        log_alpha = torch.nn.Parameter(torch.zeros([]))
        if target_entropy == 'auto':
            target_entropy = -np.prod(ac_space.shape)

    # Initialize optimizers and target networks
    loss_fn = torch.nn.MSELoss()
    pi_optim = torch.optim.Adam(policy.parameters(), lr=lr)
    qf_optim = torch.optim.Adam(
        list(q1func.parameters()) + list(q2func.parameters()), lr=lr)
    q1_targ = qf_class(vec_env, **qf_args)
    q2_targ = qf_class(vec_env, **qf_args)
    q1_targ.load_state_dict(q1func.state_dict())
    q2_targ.load_state_dict(q2func.state_dict())
    if target_entropy is not None:
        al_optim = torch.optim.Adam([log_alpha], lr=lr)

    # Save or load initial state
    if load_from is not None:
        p, idx = load_from.split(':') if ':' in load_from else (load_from, None)
        _, state = SnapshotSaver(p).get_state(idx)
        load_state_dicts(locals(), state)
        beg = state['alg']['samples']
    else:
        beg = 0
        state = dict(
            alg=dict(samples=beg),
            policy=policy.state_dict(),
            q1func=q1func.state_dict(),
            q2func=q2func.state_dict(),
            pi_optim=pi_optim.state_dict(),
            qf_optim=qf_optim.state_dict(),
            q1_targ=q1_targ.state_dict(),
            q2_targ=q2_targ.state_dict(),
            replay=replay.state_dict()
        )
        if target_entropy is not None:
            state['log_alpha'] = log_alpha
            state['al_optim'] = al_optim.state_dict()
        saver.save_state(0, state)

    # Setup and run policy tests
    ob, don = test_env.reset(), False
    @torch.no_grad()
    def test_policy():
        nonlocal ob, don
        policy.eval()
        for _ in range(10):
            while not don:
                act = policy.actions(torch.from_numpy(ob))
                ob, _, don, _ = test_env.step(act.numpy())
            don = False
        policy.train()
        log_reward_statistics(test_env, num_last_eps=10, prefix='Test')
    test_policy()
    logger.logkv("Epoch", beg // epoch)
    logger.logkv("TotalNSamples", beg)
    log_reward_statistics(vec_env)
    logger.dumpkvs()

    # Set action sampling strategies
    rand_uniform_actions = lambda _: np.stack(
        [ac_space.sample() for _ in range(vec_env.num_envs)])

    @torch.no_grad()
    def stoch_policy_actions(obs):
        return policy.actions(torch.from_numpy(obs)).numpy()

    # Algorithm main loop
    obs1 = vec_env.reset()
    prev_samp = beg
    for samples in trange(beg+1, total_samples+1, desc="Training", unit="step"):
        if samples <= start_steps:
            actions = rand_uniform_actions
        else:
            actions = stoch_policy_actions

        acts = actions(obs1)
        obs2, rews, dones, _ = vec_env.step(acts)
        as_tensors = map(
            torch.from_numpy, (obs1, acts, rews, obs2, dones.astype('f')))
        for ob1, act, rew, ob2, done in zip(*as_tensors):
            replay.store(ob1, act, rew, ob2, done)
        obs1 = obs2

        if replay.size >= mb_size:
            for _ in range(int(updates_per_step)):
                ob_1, ac_1, rew_, ob_2, done_ = replay.sample(mb_size)
                dist = policy(ob_1)
                pi_a = dist.rsample()
                logp = dist.log_prob(pi_a)
                if target_entropy is not None:
                    al_optim.zero_grad()
                    alpha_loss = torch.mean(
                        log_alpha * (logp.detach()+target_entropy)).neg()
                    alpha_loss.backward()
                    al_optim.step()
                    logger.logkv_mean("AlphaLoss", alpha_loss.item())
                    alpha = log_alpha.exp().item()

                qf_optim.zero_grad()
                with torch.no_grad():
                    pi_2 = policy(ob_2)
                    ac_2 = pi_2.sample()
                    minq = torch.min(q1_targ(ob_2, ac_2), q2_targ(ob_2, ac_2))
                    y_qf = reward_scale*rew_ \
                           + gamma*(1-done_)*(minq - alpha*pi_2.log_prob(ac_2))
                q1_val = q1func(ob_1, ac_1)
                q2_val = q2func(ob_1, ac_1)
                q1_loss = loss_fn(q1_val, y_qf).div(2)
                q2_loss = loss_fn(q2_val, y_qf).div(2)
                q1_loss.add(q2_loss).backward()
                qf_optim.step()

                pi_optim.zero_grad()
                qpi_min = torch.min(q1func(ob_1, pi_a), q2func(ob_1, pi_a))
                pi_loss = torch.mean(qpi_min - alpha*logp).neg()
                pi_loss.backward()
                pi_optim.step()

                update_polyak(q1func, q1_targ, polyak)
                update_polyak(q2func, q2_targ, polyak)

                logger.logkv_mean("Entropy", logp.mean().neg().item())
                logger.logkv_mean("Q1Val", q1_val.mean().item())
                logger.logkv_mean("Q2Val", q2_val.mean().item())
                logger.logkv_mean("QPiVal", qpi_min.mean().item())
                logger.logkv_mean("Q1Loss", q1_loss.item())
                logger.logkv_mean("Q2Loss", q2_loss.item())
                logger.logkv_mean("PiLoss", pi_loss.item())
                logger.logkv_mean("Alpha", alpha)

            prev_samp = samples

        if samples % epoch == 0:
            test_policy()
            logger.logkv("Epoch", samples // epoch)
            logger.logkv("TotalNSamples", samples)
            log_reward_statistics(vec_env)
            logger.dumpkvs()

            state = dict(
                alg=dict(samples=samples),
                policy=policy.state_dict(),
                q1func=q1func.state_dict(),
                q2func=q2func.state_dict(),
                pi_optim=pi_optim.state_dict(),
                qf_optim=qf_optim.state_dict(),
                q1_targ=q1_targ.state_dict(),
                q2_targ=q2_targ.state_dict(),
                replay=replay.state_dict()
            )
            if target_entropy is not None:
                state['log_alpha'] = log_alpha.data
                state['al_optim'] = al_optim.state_dict()
            saver.save_state(samples // epoch, state)