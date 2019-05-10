import torch
import numpy as np
from baselines import logger
from proj.utils.saver import SnapshotSaver
from proj.utils.tqdm_util import trange
from proj.utils.torch_util import update_polyak
from proj.common.models import ContinuousQFunction, ValueFunction
from proj.common.sampling import ReplayBuffer
from proj.common.log_utils import save_config, log_reward_statistics


def sac(env_maker, policy, q_func=None, val_fn=None, total_steps=int(5e5),
        gamma=0.99, replay_size=int(1e6), polyak=0.995, start_steps=10000,
        epoch=5000, mb_size=100, lr=1e-3, alpha=0.2, target_entropy=None,
        reward_scale=1.0, updates_per_step=1.0, max_ep_length=1000,
        **saver_kwargs):

    # Set and save experiment hyperparameters
    if q_func is None:
        q_func = ContinuousQFunction.from_policy(policy)
    if val_fn is None:
        val_fn = ValueFunction.from_policy(policy)
    save_config(locals())
    saver = SnapshotSaver(logger.get_dir(), locals(), **saver_kwargs)

    # Initialize environments, models and replay buffer
    vec_env = env_maker()
    test_env = env_maker(train=False)
    ob_space, ac_space = vec_env.observation_space, vec_env.action_space
    pi_class, pi_args = policy.pop('class'), policy
    qf_class, qf_args = q_func.pop('class'), q_func
    vf_class, vf_args = val_fn.pop('class'), val_fn
    policy = pi_class(vec_env, **pi_args)
    q1func = qf_class(vec_env, **qf_args)
    q2func = qf_class(vec_env, **qf_args)
    val_fn = vf_class(vec_env, **vf_args)
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
    vf_optim = torch.optim.Adam(val_fn.parameters(), lr=lr)
    vf_targ = vf_class(vec_env, **vf_args)
    vf_targ.load_state_dict(val_fn.state_dict())
    if target_entropy is not None:
        al_optim = torch.optim.Adam([log_alpha], lr=lr)

    # Save initial state
    state = dict(
        alg=dict(samples=0),
        policy=policy.state_dict(),
        q1func=q1func.state_dict(),
        q2func=q2func.state_dict(),
        val_fn=val_fn.state_dict(),
        pi_optim=pi_optim.state_dict(),
        qf_optim=qf_optim.state_dict(),
        vf_optim=vf_optim.state_dict(),
        vf_targ=vf_targ.state_dict()
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
    logger.logkv("Epoch", 0)
    logger.logkv("TotalNSamples", 0)
    log_reward_statistics(vec_env)
    logger.dumpkvs()

    # Set action sampling strategies
    rand_uniform_actions = lambda _: np.stack(
        [ac_space.sample() for _ in range(vec_env.num_envs)])

    @torch.no_grad()
    def stoch_policy_actions(obs):
        return policy.actions(torch.from_numpy(obs)).numpy()

    # Algorithm main loop
    obs1, ep_length = vec_env.reset(), 0
    for samples in trange(1, total_steps + 1, desc="Training", unit="step"):
        if samples <= start_steps:
            actions = rand_uniform_actions
        else:
            actions = stoch_policy_actions

        acts = actions(obs1)
        obs2, rews, dones, _ = vec_env.step(acts)
        ep_length += 1
        dones[0] = False if ep_length == max_ep_length else dones[0]

        as_tensors = map(
            torch.from_numpy, (obs1, acts, rews, obs2, dones.astype('f')))
        for ob1, act, rew, ob2, done in zip(*as_tensors):
            replay.store(ob1, act, rew, ob2, done)
        obs1 = obs2

        if (dones[0] or ep_length == max_ep_length) and replay.size >= mb_size:
            for _ in range(int(ep_length * updates_per_step)):
                ob_1, act_, rew_, ob_2, done_ = replay.sample(mb_size)
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

                with torch.no_grad():
                    y_qf = reward_scale*rew_ + gamma*(1-done_)*vf_targ(ob_2)
                    y_vf = torch.min(q1func(ob_1, pi_a), q2func(ob_1, pi_a)) \
                           - alpha*logp

                qf_optim.zero_grad()
                q1_val = q1func(ob_1, act_)
                q2_val = q2func(ob_1, act_)
                q1_loss = loss_fn(q1_val, y_qf).div(2)
                q2_loss = loss_fn(q2_val, y_qf).div(2)
                q1_loss.add(q2_loss).backward()
                qf_optim.step()

                vf_optim.zero_grad()
                vf_val = val_fn(ob_1)
                vf_loss = loss_fn(vf_val, y_vf).div(2)
                vf_loss.backward()
                vf_optim.step()

                pi_optim.zero_grad()
                qpi_val = q1func(ob_1, pi_a)
                # qpi_val = torch.min(q1func(ob_1, pi_a), q2func(ob_1, pi_a))
                pi_loss = qpi_val.sub(logp, alpha=alpha).mean().neg()
                pi_loss.backward()
                pi_optim.step()

                update_polyak(val_fn, vf_targ, polyak)

                logger.logkv_mean("Entropy", logp.mean().neg().item())
                logger.logkv_mean("Q1Val", q1_val.mean().item())
                logger.logkv_mean("Q2Val", q2_val.mean().item())
                logger.logkv_mean("VFVal", vf_val.mean().item())
                logger.logkv_mean("QPiVal", qpi_val.mean().item())
                logger.logkv_mean("Q1Loss", q1_loss.item())
                logger.logkv_mean("Q2Loss", q2_loss.item())
                logger.logkv_mean("VFLoss", vf_loss.item())
                logger.logkv_mean("PiLoss", pi_loss.item())
                logger.logkv_mean("Alpha", alpha)

            ep_length = 0

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
                val_fn=val_fn.state_dict(),
                pi_optim=pi_optim.state_dict(),
                qf_optim=qf_optim.state_dict(),
                vf_optim=vf_optim.state_dict(),
                vf_targ=vf_targ.state_dict()
            )
            if target_entropy is not None:
                state['log_alpha'] = log_alpha
                state['al_optim'] = al_optim.state_dict()
            saver.save_state(samples // epoch, state)
