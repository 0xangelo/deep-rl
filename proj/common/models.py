import math
import torch
import torch.nn as nn
import numpy as np
import gym.spaces as spaces
from abc import ABC, abstractmethod
from proj.common import distributions


class ToFloat(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.scale = 1 / 250.0 if dtype is np.uint8 else 1

    def forward(self, x):
        return x.float() * self.scale


class OneHot(nn.Module):
    def __init__(self, n_cat):
        super().__init__()
        self.n_cat = n_cat

    def forward(self, x):
        return torch.eye(n_cat)[x]


class Flatten(nn.Module):
    def __init__(self, flat_size):
        super().__init__()
        self.flat_size = flat_size

    def forward(self, x):
        return x.reshape(-1, self.flat_size)

# ==============================
# Models
# ==============================

class Model(nn.Module):
    def __init__(self, env, **kwargs):
        super().__init__()
        self.ob_space = space = env.observation_space
        if isinstance(space, spaces.Box):
            self.in_features = np.prod(space.shape)
            self.process_obs = ToFloat(space.dtype.type)
        elif isinstance(space, spaces.Discrete):
            self.in_features = space.n
            self.process_obs = OneHot(space.n)
        else:
            raise ValueError("{} is not a valid space type".format(str(space)))

    def forward(self, obs):
        return self.process_obs(obs)


class FeedForwardModel(ABC):
    out_features = None

    @abstractmethod
    def forward(self, obs):
        pass


class MlpModel(Model, FeedForwardModel):
    def __init__(self, env, *, hidden_sizes=[32, 32], activation=nn.Tanh,
                 **kwargs):
        super().__init__(env, **kwargs)
        if isinstance(activation, str):
            if activation == 'tanh':
                activation = nn.Tanh
            elif activation == 'relu':
                activation = nn.ReLU
            elif activation == 'elu':
                activation = nn.ELU
            else:
                raise ValueError(
                    "Invalid string option '{}' for activation".format(
                        activation
                    ))
        self.activation = activation
        self.hidden_sizes = hidden_sizes

        layers, in_sizes = [], [self.in_features] + hidden_sizes
        for in_features, out_features in zip(in_sizes[:-1], in_sizes[1:]):
            layers.append(nn.Linear(in_features, out_features))
            layers.append(activation())
        self.hidden_net = nn.Sequential(*layers)
        self.out_features = in_sizes[-1]

        def initialize(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.constant_(module.bias, 0)
        self.hidden_net.apply(initialize)

    def forward(self, obs):
        obs = super().forward(obs)
        return self.hidden_net(obs)


class CNNModel(Model, FeedForwardModel):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.in_channels = self.ob_space.shape[-1]
        self.hidden_net = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1),
            nn.ReLU(),
            Flatten(32 * 7 * 7),
            nn.Linear(32 * 7 * 7, 512),
            nn.ReLU()
        )
        self.out_features = 512

        def initialize(module):
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(
                    module.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(module.bias, 0)
        self.hidden_net.apply(initialize)

    def forward(self, obs):
        obs = super().forward(obs)
        # Add a batch dim if necessary and move channels to appropriate pos
        obs = obs.reshape(-1, *self.ob_space.shape)
        obs = obs.permute(0, 3, 1, 2)
        feats = self.hidden_net(obs)
        # Remove batch dim for single observations
        return feats.squeeze(0)

# ==============================
# Policies
# ==============================

class Policy(ABC):
    ac_space = None

    @abstractmethod
    def forward(self, obs):
        """
        Given a batch of observations, return a batch of corresponding
        action distributions.

        Args:
        obs (Tensor): A batch of observations

        return (proj.common.Distribution): A batch of action distributions
        """
        pass

    def actions(self, obs):
        """
        Given a batch of observations, return a batch of actions,
        each sampled from the corresponding action distributions.

        Args:
        obs (Tensor): A batch of observations

        return (Tensor): A batch of actions
        """
        return self(obs).sample()


class FeedForwardPolicy(FeedForwardModel, Policy):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.ac_space = env.action_space
        self.pdtype = distributions.pdtype(self.ac_space, self.out_features)

    def forward(self, obs):
        return self.pdtype(super().forward(obs))


class MlpPolicy(FeedForwardPolicy, MlpModel):
    pass


class CNNPolicy(FeedForwardPolicy, CNNModel):
    pass

# ==============================
# Value Functions
# ==============================

class ValueFunction(ABC):
    @abstractmethod
    def forward(self, obs):
        """
        Given some observations, compute the state values.

        Args:
            obs (Tensor): A batch of observations

        return (Tensor): state values for each observation
        """
        pass

    @staticmethod
    def from_policy(policy):
        pol_type = policy['class']
        if issubclass(pol_type, MlpModel):
            kwargs = {'class': MlpValueFunction, 'activation': nn.ELU}
            if 'hidden_sizes' in policy:
                kwargs['hidden_sizes'] = policy['hidden_sizes']
            return kwargs
        elif issubclass(pol_type, CNNModel):
            return {'class': CNNValueFunction}
        else:
            raise ValueError("Unrecognized policy type")


class ZeroValueFunction(Model, ValueFunction):
    def forward(self, obs):
        return torch.zeros(len(obs.reshape(-1, *self.ob_space.shape)))


class FeedForwardValueFunction(FeedForwardModel, ValueFunction):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.val_layer = nn.Linear(self.out_features, 1)
        nn.init.orthogonal_(self.val_layer.weight, gain=0.01)

    def forward(self, obs):
        feats = super().forward(obs)
        return self.val_layer(feats).squeeze()


class MlpValueFunction(FeedForwardValueFunction, MlpModel):
    pass


class CNNValueFunction(FeedForwardValueFunction, CNNModel):
    pass

# ==============================
# Weight sharing models
# ==============================

class WeightSharingAC(ABC):
    ac_space = None

    @abstractmethod
    def forward(self, obs):
        """
        Given some observations, compute the corresponding
        action distributions and state values.

        Args:
            obs (Tensor): A batch of observations

        return (Distribution, Tensor): action distributions and state values
        """
        pass

    def actions(self, obs):
        """
        Given a batch of observations, return a batch of actions,
        each sampled from the corresponding action distributions.

        Args:
        obs (Tensor): A batch of observations

        return (Tensor): A batch of actions
        """
        return self(obs)[0].sample()


class FeedForwardWeightSharingAC(FeedForwardModel, WeightSharingAC):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.ac_space = env.action_space
        self.pdtype = distributions.pdtype(self.ac_space, self.out_features)
        self.val_layer = nn.Linear(self.out_features, 1)
        nn.init.orthogonal_(self.val_layer.weight, gain=1.0)
        nn.init.constant_(self.val_layer.bias, 0)

    def forward(self, obs):
        feats = super().forward(obs)
        return self.pdtype(feats), self.val_layer(feats).squeeze()


class MlpWeightSharingAC(FeedForwardWeightSharingAC, MlpModel):
    pass


class CNNWeightSharingAC(FeedForwardWeightSharingAC, CNNModel):
    pass
