import math
import numpy as np
from gym import logger
from gym.envs.classic_control.cartpole import CartPoleEnv


class RandomCartPoleEnv(CartPoleEnv):
    def __init__(self, *args, noise_scale=0.1, **kwargs):
        self.noise_scale = noise_scale
        super().__init__(*args, **kwargs)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )
        x_pos, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        # === New code: add gaussian noise ===
        force += force * self.np_random.normal(scale=self.noise_scale)

        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (
            force + self.polemass_length * theta_dot * theta_dot * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)
        )
        x_acc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        if self.kinematics_integrator == "euler":
            x_pos = x_pos + self.tau * x_dot
            x_dot = x_dot + self.tau * x_acc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * x_acc
            x_pos = x_pos + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        self.state = (x_pos, x_dot, theta, theta_dot)
        done = bool(
            x_pos < -self.x_threshold
            or x_pos > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has "
                    "already returned done = True. You should always call 'reset()' "
                    "once you receive 'done = True' -- any further steps are "
                    "undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}
