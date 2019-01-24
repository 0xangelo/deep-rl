import math, torch, torch.nn as nn, numpy as np, gym.spaces as spaces
from abc import ABC, abstractmethod
from proj.common import distributions


# ==============================
# Models
# ==============================

class Model(nn.Module):
    def __init__(self, env, **kwargs):
        super().__init__()
        self.ob_space = space = env.observation_space
        if isinstance(space, spaces.Box):
            n_features = np.prod(space.shape)
        elif isinstance(space, spaces.Discrete):
            n_features = space.n # if space.n > 2 else 1
        else:
            raise ValueError("{} is not a valid space type".format(str(space)))
        self.in_features = n_features


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
        return self.hidden_net(obs)


class CNNModel(Model, FeedForwardModel):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.in_channels = self.ob_space.shape[-1]
        self.conv1 = nn.Conv2d(self.in_channels, 16, 8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
        self.fc = nn.Linear(2592, 256)
        self.out_features = 256

        def initialize(*modules):
            for module in modules:
                if isinstance(module, nn.Conv2d):
                    nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                    nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    nn.init.constant_(module.bias, 0)
        initialize(self.conv1, self.conv2, self.fc)

    def forward(self, obs):
        obs = obs.transpose(1, 2)
        obs = obs.transpose(1, 3)
        relu = nn.ReLU()
        h1 = relu(self.conv1(obs))
        h2 = relu(self.conv2(h1))
        h3 = relu(self.fc(h2.view(len(obs), -1)))
        return h3


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


class ZeroBaseline(Model, Baseline):
    def forward(self, obs):
        return torch.zeros(len(obs))


class FeedForwardBaseline(FeedForwardModel, Baseline):
    def __init__(self, env, **kwargs):
        # For input, we will concatenate the observation with the time, so we
        # need to increment the observation dimension
        ob_space = env.observation_space
        if isinstance(ob_space, spaces.Box) and len(ob_space.shape) == 1:
            self.concat_time = True
            # Use environment to send new ob_space to superclass
            env.observation_space = spaces.Box(
                low=np.append(ob_space.low, 0),
                high=np.append(ob_space.high, 2 ** 32),
            )
            self.timestep_limit = env.spec.timestep_limit
        super().__init__(env, **kwargs)
        # Restore original ob_space if changed
        env.observation_space = ob_space

        self.val_layer = nn.Linear(self.out_features, 1)
        nn.init.orthogonal_(self.val_layer.weight, gain=0.01)

    def forward(self, obs):
        if self.concat_time:
            # Append relative timestep to observation to properly identify state
            ts = torch.arange(len(obs), dtype=torch.get_default_dtype()) \
                                               / self.timestep_limit
            obs = torch.cat((obs, ts[:, None]), dim=-1)
        feats = super().forward(obs)
        return torch.squeeze(self.val_layer(feats))


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
        nn.init.orthogonal_(self.val_layer.weight, gain=0.01)

    def forward(self, obs):
        feats = super().forward(obs)
        return self.pdtype(feats), self.val_layer(feats).squeeze()


class MlpWeightSharingAC(FeedForwardWeightSharingAC, MlpModel):
    pass


class CNNWeightSharingAC(FeedForwardWeightSharingAC, CNNModel):
    pass
