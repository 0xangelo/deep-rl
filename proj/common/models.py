import math, torch, torch.nn as nn
from abc import ABC, abstractmethod
from proj.common import distributions
from proj.common.utils import n_features


# ==============================
# Models
# ==============================

class Model(nn.Module):
    def __init__(self, env, **kwargs):
        super().__init__()
        self.ob_space = env.observation_space
        self.in_features = n_features(self.ob_space)


class FeedForwardModel(ABC):
    @abstractmethod
    def build_hidden_net(self):
        pass

    @abstractmethod
    def compute_features(self, obs):
        pass

    @staticmethod
    @abstractmethod
    def initialize(module):
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

    def build_hidden_net(self):
        assert not hasattr(self, 'hidden_net'), \
            "Can only call 'build_hidden_net' once."
        layers, in_sizes = [], [self.in_features] + self.hidden_sizes
        for in_features, out_features in zip(in_sizes[:-1], in_sizes[1:]):
            layers.append(nn.Linear(in_features, out_features))
            layers.append(self.activation())

        self.hidden_net = nn.Sequential(*layers)
        self.out_features = in_sizes[-1]

    def compute_features(self, obs):
        return self.hidden_net(obs)

    @staticmethod
    def initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            nn.init.constant_(module.bias, 0)


class CNNModel(Model, FeedForwardModel):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.in_channels = self.ob_space.shape[-1]

    def build_hidden_net(self):
        self.conv1 = nn.Conv2d(self.in_channels, 16, 8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
        self.fc = nn.Linear(2592, 256)
        self.out_features = 256

    def compute_features(self, obs):
        obs = obs.transpose(1, 2)
        obs = obs.transpose(1, 3)
        h1 = nn.functional.relu(self.conv1(obs))
        h2 = nn.functional.relu(self.conv2(h1))
        h3 = nn.functional.relu(self.fc(h3))
        return h3

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            nn.init.constant_(module.bias, 0)


# ==============================
# Policies
# ==============================

class Policy(ABC):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.ac_space = env.action_space

    @abstractmethod
    def forward(self, obs):
        """
        Given some observations, returns the parameters
        for the action distributions.

        Args:
        obs (Tensor): A batch of observations

        return (Tensor): distribution parameters
        """
        pass

    @abstractmethod
    def dists(self, obs):
        """
        Given a batch of observations, return a batch of corresponding
        action distributions.

        Args:
        obs (Tensor): A batch of observations

        return (proj.common.Distribution): A batch of action distributions
        """
        pass

    @torch.no_grad()
    def actions(self, obs):
        """
        Given a batch of observations, return a batch of actions,
        each sampled from the corresponding action distributions.

        Args:
        obs (Tensor): A batch of observations

        return (Tensor): A batch of actions
        """
        return self.dists(obs).sample()

    def action(self, ob):
        """
        Single observation version of Policy.actions,
        with batch dimension removed.
        """
        return self.actions(ob.unsqueeze(0))[0]


class FeedForwardPolicy(Policy, FeedForwardModel):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.build_hidden_net()
        self.apply(self.initialize)

        self.pdtype = distributions.pdtype(self.ac_space, self.out_features)

    def forward(self, obs):
        feats = self.compute_features(obs)
        return self.pdtype(feats)

    def dists(self, obs):
        return self.pdtype.from_flat(self(obs))


class MlpPolicy(FeedForwardPolicy, MlpModel):
    pass


class CNNPolicy(FeedForwardPolicy, CNNModel):
    pass


# ==============================
# Baselines
# ==============================

class Baseline(ABC):
    @abstractmethod
    def forward(self, obs):
        """
        Given some observations, compute the baselines.

        Args:
            obs (Tensor): A batch of observations

        return (Tensor): baselines for each observation
        """
        pass


class ZeroBaseline(Baseline, Model):
    def forward(self, obs):
        return torch.zeros(len(obs))


class FeedForwardBaseline(Baseline, FeedForwardModel):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.timestep_limit = env.spec.timestep_limit
        self.in_features += 1
        self.build_hidden_net()
        self.apply(self.initialize)

        self.val_layer = nn.Linear(self.out_features, 1)
        nn.init.orthogonal_(self.val_layer.weight, gain=0.01)

    def forward(self, obs):
        # Append relative timestep to observation to properly identify state
        ts = torch.arange(len(obs), dtype=torch.get_default_dtype()) \
                                         / self.timestep_limit
        h = self.compute_features(torch.cat((obs, ts[:, None]), dim=-1))
        return torch.squeeze(self.val_layer(h))


class MlpBaseline(FeedForwardBaseline, MlpModel):
    pass


class CNNBaseline(FeedForwardBaseline, CNNModel):
    pass


def default_baseline(policy):
    pol_type = policy['class']
    if issubclass(pol_type, MlpModel):
        kwargs = {'class': MlpBaseline, 'activation': nn.ELU}
        if 'hidden_sizes' in policy:
            kwargs['hidden_sizes'] = policy['hidden_sizes']
        return kwargs
    elif issubclass(pol_type, CNNModel):
        return {'class': CNNBaseline}
    else:
        raise ValueError("Unrecognized policy type")


# ==============================
# Weight sharing models
# ==============================

class WeightSharingAC(ABC):
    @abstractmethod
    def dists_values(self, obs):
        """
        Given some observations, compute the corresponding
        action distributions and state values.

        Args:
            obs (Tensor): A batch of observations

        return (Distribution, Tensor): action distributions and state values
        """
        pass


class FeedForwardWeightSharingAC(WeightSharingAC, FeedForwardPolicy):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.val_layer = nn.Linear(self.out_features, 1)
        nn.init.orthogonal_(self.val_layer.weight, gain=0.01)

    def dists_values(self, obs):
        feats = self.compute_features(obs)
        dists = self.pdtype.from_flat(self.pdtype(feats))
        values = torch.squeeze(self.val_layer(feats))
        return dists, values


class MlpWeightSharingAC(FeedForwardWeightSharingAC, MlpPolicy):
    pass


class CNNWeightSharingAC(FeedForwardWeightSharingAC, CNNPolicy):
    pass
