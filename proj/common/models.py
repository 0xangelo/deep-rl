"""
Collection of neural network models, policies and value functions in PyTorch.
"""
import math
import torch
import torch.nn as nn
import numpy as np
import gym.spaces as spaces
import proj.common.distributions as dists
from abc import ABC, abstractmethod
from proj.utils.torch_util import ToFloat, Concat, OneHot, Flatten


# ==============================
# Models
# ==============================
class Model(nn.Module):
    def __init__(self, env, concat_action=False, **kwargs):
        super().__init__()
        self.ob_space = ob_space = env.observation_space
        if concat_action:
            self.ac_space = ac_space = env.action_space
            assert all(
                (
                    isinstance(ac_space, spaces.Box),
                    len(ac_space.shape) == 1,
                    isinstance(ob_space, spaces.Box),
                    len(ob_space.shape) == 1,
                )
            ), "Currently only supports concatenating continuous spaces"
            self.in_features = np.prod(ob_space.shape) + np.prod(ac_space.shape)
            self.process_input = Concat(ob_space.dtype.type)
        elif isinstance(ob_space, spaces.Box):
            self.in_features = np.prod(ob_space.shape)
            self.process_input = ToFloat(ob_space.dtype.type)
        elif isinstance(ob_space, spaces.Discrete):
            self.in_features = ob_space.n
            self.process_input = OneHot(ob_space.n)
        else:
            raise ValueError("{} is not a valid space type".format(str(ob_space)))

    def forward(self, *args):
        return self.process_input(*args)


class FeedForwardModel(ABC):
    out_features = None

    @abstractmethod
    def forward(self, obs):
        pass


class MlpModel(Model, FeedForwardModel):
    def __init__(self, env, *, hidden_sizes=(32, 32), activation="elu", **kwargs):
        super().__init__(env, **kwargs)
        if isinstance(activation, str):
            if activation == "tanh":
                activation = nn.Tanh
            elif activation == "relu":
                activation = nn.ReLU
            elif activation == "elu":
                activation = nn.ELU
            else:
                raise ValueError(
                    "Invalid string option '{}' for activation".format(activation)
                )
        self.activation = activation
        self.hidden_sizes = hidden_sizes = list(hidden_sizes)

        layers, in_sizes = [], [self.in_features] + hidden_sizes
        for in_features, out_features in zip(in_sizes[:-1], in_sizes[1:]):
            layers.append(nn.Linear(in_features, out_features))
            layers.append(activation())
        if len(self.ob_space.shape) > 1:
            layers = [Flatten(self.in_features)] + layers
        self.hidden_net = nn.Sequential(*layers)
        self.out_features = in_sizes[-1]

        def initialize(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.constant_(module.bias, 0)

        self.hidden_net.apply(initialize)

    def forward(self, *args):
        processed_input = super().forward(*args)
        return self.hidden_net(processed_input)


class CNNModel(Model, FeedForwardModel):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.in_channels = self.ob_space.shape[-1]
        self.hidden_net = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2),
            nn.ReLU(),
            Flatten(2592),
            nn.Linear(2592, 256),
            nn.ReLU(),
        )
        self.out_features = 256

        def initialize(module):
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
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

        return self(obs).sample() if self.training else self(obs).mode


class FeedForwardPolicy(FeedForwardModel, Policy):
    def __init__(self, env, clamp_acts=False, indep_std=True, **kwargs):
        super().__init__(env, **kwargs)
        self.ac_space = env.action_space
        self.pdtype = dists.pdtype(
            self.ac_space, self.out_features, clamp_acts=clamp_acts, indep_std=indep_std
        )

    def forward(self, obs):
        return self.pdtype(super().forward(obs))


class MlpPolicy(FeedForwardPolicy, MlpModel):
    pass


class CNNPolicy(FeedForwardPolicy, CNNModel):
    pass


class DeterministicPolicy(ABC):
    """
    Limited to continuous action spaces
    """

    ac_space = None

    @abstractmethod
    def forward(self, obs):
        """
        Given a batch of observations, return a batch of actions

        Args:
        obs (Tensor): A batch of observations

        return (Tensor): A batch of actions
        """
        pass

    def actions(self, obs):
        """
        Given a batch of observations, return a batch of actions.

        Args:
        obs (Tensor): A batch of observations

        return (Tensor): A batch of actions
        """
        return self(obs)


class FeedForwardDeterministicPolicy(FeedForwardModel, DeterministicPolicy):
    def __init__(self, env, **kwargs):
        assert isinstance(
            env.action_space, spaces.Box
        ), "Deterministic policies only handle continuous action spaces"
        super().__init__(env, **kwargs)
        self.ac_space = env.action_space
        self.act_layer = nn.Linear(self.out_features, self.ac_space.shape[-1])
        self.act_activ = nn.Tanh()
        low, high = map(torch.Tensor, (self.ac_space.low, self.ac_space.high))
        self.loc = (high + low) / 2
        self.scale = (high - low) / 2

    def forward(self, obs):
        feats = super().forward(obs)
        tanh = self.act_activ(self.act_layer(feats))
        return self.loc + tanh * self.scale


class MlpDeterministicPolicy(FeedForwardDeterministicPolicy, MlpModel):
    pass


class CNNDeterministicPolicy(FeedForwardDeterministicPolicy, CNNModel):
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
        pol_type = policy["class"]
        if issubclass(pol_type, MlpModel):
            kwargs = {"class": MlpValueFunction, "activation": nn.ELU}
            if "hidden_sizes" in policy:
                kwargs["hidden_sizes"] = policy["hidden_sizes"]
            return kwargs
        elif issubclass(pol_type, CNNModel):
            return {"class": CNNValueFunction}
        else:
            raise ValueError("Unrecognized policy type")


class ZeroValueFunction(Model, ValueFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dummy = nn.Parameter(torch.zeros([]))

    def forward(self, obs):
        batch_dims = obs.shape[: len(obs.shape) - len(self.ob_space.shape)]
        return torch.zeros(batch_dims, requires_grad=True)


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


class ContinuousQFunction(ABC):
    @abstractmethod
    def forward(self, obs, acts):
        """
        Given observation-action pairs, computes the estimated
        optimal Q-values.
        """
        pass

    @staticmethod
    def from_policy(policy):
        pol_type = policy["class"]
        if issubclass(pol_type, MlpModel):
            kwargs = {"class": MlpContinuousQFunction, "activation": nn.ReLU}
            if "hidden_sizes" in policy:
                kwargs["hidden_sizes"] = policy["hidden_sizes"]
            return kwargs
        else:
            raise ValueError("Unrecognized policy type")


class FeedForwardContinuousQFunction(FeedForwardModel, ContinuousQFunction):
    def __init__(self, env, **kwargs):
        super().__init__(env, concat_action=True, **kwargs)
        self.val_layer = nn.Linear(self.out_features, 1)
        nn.init.orthogonal_(self.val_layer.weight, gain=0.01)

    def forward(self, obs, acts):
        feats = super().forward(obs, acts)
        return self.val_layer(feats).squeeze()


class MlpContinuousQFunction(FeedForwardContinuousQFunction, MlpModel):
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
        self.pdtype = dists.pdtype(self.ac_space, self.out_features)
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
