"""
Cart pole swing-up: modified version of:
https://github.com/hardmaru/estool/blob/master/custom_envs/cartpole_swingup.py
"""
import math
from collections import namedtuple

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering


Physics = namedtuple(
    "Physics", "gravity masscart masspole polelen forcemag deltat friction"
)

State = namedtuple("State", "x_pos x_dot theta theta_dot")

Screen = namedtuple("Screen", "width height world_width polewidth cartwidth cartheight")


class CartPoleSwingUpEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}
    physics = Physics(
        gravity=9.82,  # gravity
        masscart=0.5,  # cart mass
        masspole=0.5,  # pendulum mass
        polelen=0.6,  # pole's length
        forcemag=10.0,
        deltat=0.01,  # seconds between state updates
        friction=0.1,  # friction coefficient
    )
    thresholds = {
        "theta_radians": 12 * 2 * math.pi / 360,  # Angle at which to fail the episode
        "x_pos": 2.4,  # Distance limit from the center
    }

    def __init__(self):
        high = np.array(
            [
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
            ]
        )

        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,))
        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # Valid action
        action = np.clip(action, -1.0, 1.0)[0]
        action *= self.physics.forcemag

        state = self.state
        physics = self.physics

        sin_theta = math.sin(state.theta)
        cos_theta = math.cos(state.theta)

        m_p_l = physics.masspole * physics.polelen
        masstotal = physics.masscart + physics.masspole
        xdot_update = (
            -2 * m_p_l * (state.theta_dot ** 2) * sin_theta
            + 3 * physics.masspole * physics.gravity * sin_theta * cos_theta
            + 4 * action
            - 4 * physics.friction * state.x_dot
        ) / (4 * masstotal - 3 * physics.masspole * cos_theta ** 2)
        thetadot_update = (
            -3 * m_p_l * (state.theta_dot ** 2) * sin_theta * cos_theta
            + 6 * masstotal * physics.gravity * sin_theta
            + 6 * (action - physics.friction * state.x_dot) * cos_theta
        ) / (4 * physics.polelen * masstotal - 3 * m_p_l * cos_theta ** 2)

        self.state = state = State(
            x_pos=state.x_pos + state.x_dot * physics.deltat,
            theta=state.theta + state.theta_dot * physics.deltat,
            x_dot=state.x_dot + xdot_update * physics.deltat,
            theta_dot=state.theta_dot + thetadot_update * physics.deltat,
        )

        done = False
        if (
            state.x_pos < -self.thresholds["x_pos"]
            or state.x_pos > self.thresholds["x_pos"]
        ):
            done = True

        reward_theta = (np.cos(state.theta) + 1.0) / 2.0
        reward_x = np.cos((state.x_pos / self.thresholds["x_pos"]) * (np.pi / 2.0))
        reward = reward_theta * reward_x

        return self._get_obs(), reward, done, {}

    def reset(self):
        self.state = State(
            *np.random.normal(
                loc=np.array([0.0, 0.0, np.pi, 0.0]),
                scale=np.array([0.2, 0.2, 0.2, 0.2]),
            )
        )
        return self._get_obs()

    def render(self, mode="human"):
        if self.viewer is None:
            self.viewer = CartPoleSwingUpViewer(self.physics)

        if self.state is None:
            return None

        self.viewer.update(self.state, self.physics)
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _get_obs(self):
        state = self.state
        return np.array(
            [
                state.x_pos,
                state.x_dot,
                np.cos(state.theta),
                np.sin(state.theta),
                state.theta_dot,
            ]
        )


class CartPoleSwingUpViewer:
    screen = Screen(
        width=600,
        height=400,  # before was 600
        world_width=5,  # max visible position of cart
        polewidth=6.0,
        cartwidth=40.0,
        cartheight=20.0,
    )

    def __init__(self, physics):
        screen = self.screen
        self.viewer = rendering.Viewer(screen.width, screen.height)
        self.transforms = {
            "cart": rendering.Transform(),
            "pole": rendering.Transform(translation=(0, 0)),
            "pole_bob": rendering.Transform(),
            "wheel_l": rendering.Transform(
                translation=(-screen.cartwidth / 2, -screen.cartheight / 2)
            ),
            "wheel_r": rendering.Transform(
                translation=(screen.cartwidth / 2, -screen.cartheight / 2)
            ),
        }

        self._init_track()
        self._init_cart()
        self._init_pole(physics)
        self._init_axle()
        # Make another circle on the top of the pole
        self._init_pole_bob()
        self._init_wheels()

    def _init_track(self):
        screen = self.screen
        carty = screen.height / 2
        track_height = carty - screen.cartheight / 2 - screen.cartheight / 4
        track = rendering.Line((0, track_height), (screen.width, track_height))
        track.set_color(0, 0, 0)
        self.viewer.add_geom(track)

    def _init_cart(self):
        screen = self.screen
        lef, rig, top, bot = (
            -screen.cartwidth / 2,
            screen.cartwidth / 2,
            screen.cartheight / 2,
            -screen.cartheight / 2,
        )
        cart = rendering.FilledPolygon([(lef, bot), (lef, top), (rig, top), (rig, bot)])
        cart.add_attr(self.transforms["cart"])
        cart.set_color(1, 0, 0)
        self.viewer.add_geom(cart)

    def _init_pole(self, physics):
        screen = self.screen
        scale = screen.width / screen.world_width
        lef, rig, top, bot = (
            -screen.polewidth / 2,
            screen.polewidth / 2,
            scale * physics.polelen - screen.polewidth / 2,
            -screen.polewidth / 2,
        )
        pole = rendering.FilledPolygon([(lef, bot), (lef, top), (rig, top), (rig, bot)])
        pole.set_color(0, 0, 1)
        pole.add_attr(self.transforms["pole"])
        pole.add_attr(self.transforms["cart"])
        self.viewer.add_geom(pole)

    def _init_axle(self):
        axle = rendering.make_circle(self.screen.polewidth / 2)
        axle.add_attr(self.transforms["pole"])
        axle.add_attr(self.transforms["cart"])
        axle.set_color(0.1, 1, 1)
        self.viewer.add_geom(axle)

    def _init_pole_bob(self):
        pole_bob = rendering.make_circle(self.screen.polewidth / 2)
        pole_bob.add_attr(self.transforms["pole_bob"])
        pole_bob.add_attr(self.transforms["pole"])
        pole_bob.add_attr(self.transforms["cart"])
        pole_bob.set_color(0, 0, 0)
        self.viewer.add_geom(pole_bob)

    def _init_wheels(self):
        screen = self.screen
        wheel_l = rendering.make_circle(screen.cartheight / 4)
        wheel_r = rendering.make_circle(screen.cartheight / 4)
        wheel_l.add_attr(self.transforms["wheel_l"])
        wheel_l.add_attr(self.transforms["cart"])
        wheel_r.add_attr(self.transforms["wheel_r"])
        wheel_r.add_attr(self.transforms["cart"])
        wheel_l.set_color(0, 0, 0)  # Black, (B, G, R)
        wheel_r.set_color(0, 0, 0)  # Black, (B, G, R)
        self.viewer.add_geom(wheel_l)
        self.viewer.add_geom(wheel_r)

    def update(self, state, physics):
        screen = self.screen
        scale = screen.width / screen.world_width

        cartx = state.x_pos * scale + screen.width / 2.0  # MIDDLE OF CART
        carty = screen.height / 2
        self.transforms["cart"].set_translation(cartx, carty)
        self.transforms["pole"].set_rotation(state.theta)
        self.transforms["pole_bob"].set_translation(
            -physics.polelen * np.sin(state.theta),
            physics.polelen * np.cos(state.theta),
        )

    def render(self, *args, **kwargs):
        self.viewer.render(*args, **kwargs)

    def close(self):
        self.viewer.close()
