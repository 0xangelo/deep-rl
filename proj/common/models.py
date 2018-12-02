import math, torch, torch.nn as nn
from abc import ABC, abstractmethod
from . import distributions
from .observations import n_features
from gym import spaces

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
    def build_network(self):
        pass

    @staticmethod
    @abstractmethod
    def initialize(module):
        pass


class MlpModel(Model, FeedForwardModel):
    def __init__(self, env, *, hidden_sizes=[64, 64], activation=nn.Tanh,
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

    def build_network(self):
        layers, in_sizes = [], [self.in_features] + self.hidden_sizes
        for in_features, out_features in zip(in_sizes[:-1], in_sizes[1:]):
            layers.append(nn.Linear(in_features, out_features))
            layers.append(self.activation())
        layers.append(nn.Linear(in_sizes[-1], self.out_features))
            
        return nn.Sequential(*layers) 

    @staticmethod
    def initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            nn.init.constant_(module.bias, 0)        


# ==============================
# Policies
# ==============================

class AbstractPolicy(ABC):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.ac_space = env.action_space
        self.out_features = n_features(self.ac_space)
        self.make_pd = distributions.pd_maker(self.ac_space, self)
        
    @abstractmethod
    def actions(self, obs):
        """
        Given a batch of observations, return a batch of actions, 
        each sampled from the corresponding action distributions.

        Args:
        x (Tensor): A batch of observations

        return (Tensor): A batch of actions
        """
        pass

    @abstractmethod
    def forward(self, x):
        """
        Given some observations, returns the parameters 
        for the action distributions.

        Args:
        x (Tensor): A batch of observations

        return (Tensor): distribution parameters
        """
        pass

    @abstractmethod
    def action(self, ob):
        """
        Single observation version of AbstractPolicy.actions,
        with batch dimension removed.
        """
        pass

    @abstractmethod
    def dists(self, obs):
        """
        Given a batch of observations, return a batch of corresponding 
        action distributions.

        Args:
        obs (Tensor): A batch of observations

        return (Tensor): A batch of action distributions
        """
        pass


class FeedForwardPolicy(AbstractPolicy, FeedForwardModel):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.network = self.build_network()
        
        if isinstance(self.ac_space, spaces.Box):
            self.logstd = nn.Parameter(torch.zeros(self.out_features))

        self.apply(self.initialize)
        nn.init.orthogonal_(self.network[-1].weight, gain=0.01)
        nn.init.constant_(self.network[-1].bias, 0)        
        
    def forward(self, x):
        return self.network(x)

    @torch.no_grad()
    def actions(self, obs):
        return self.dists(obs).sample()

    def action(self, ob):
        return self.actions(ob.unsqueeze(0))[0]

    def dists(self, obs):
        return self.make_pd(self(obs))
        

class MlpPolicy(FeedForwardPolicy, MlpModel):
    pass


# ==============================
# Baselines
# ==============================

class AbstractBaseline(ABC):
    @abstractmethod
    def forward(self, x):
        """
        Given some observations, compute the baselines.
        
        Args:
            x (Tensor): A batch of observations

        return (Tensor): baselines for each observation
        """
        pass

    @abstractmethod
    def update(self, buffer):
        """
        Given a buffer with observations, returns and baselines,
        update the reinforcement baseline.

        Args:
        buffer (dict): A buffer containing 'observations', 'returns' and 
        'baselines' as keys.
        """
        pass


class ZeroBaseline(AbstractBaseline, Model):
    def forward(self, x):
        return torch.zeros(len(x))

    def update(self, buffer):
        pass


class FeedForwardBaseline(AbstractBaseline, FeedForwardModel):
    def __init__(self, env, *, mixture_fraction=0.1, **kwargs):
        super().__init__(env, **kwargs)
        self.mixture_fraction = mixture_fraction
        self.timestep_limit = env.spec.timestep_limit
        self.in_features += 1
        self.out_features = 1
        self.network = self.build_network()
        self.apply(self.initialize)

    def forward(self, x):
        # Append relative timestep to observation to properly identify state
        ts = torch.arange(len(x), dtype=torch.get_default_dtype()) \
                                         / self.timestep_limit
        return torch.squeeze(self.network(torch.cat((x, ts[:, None]), dim=-1)))

    def update(self, buffer):
        observations = buffer["observations"]
        targets = self.mixture_fraction * buffer["baselines"] + \
                  (1 - self.mixture_fraction) * buffer["returns"]

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.LBFGS(self.parameters(), max_iter=10)
        def closure():
            optimizer.zero_grad()
            loss = loss_fn(self(observations), targets)
            loss.backward()
            return loss

        optimizer.step(closure)


class MlpBaseline(FeedForwardBaseline, MlpModel):
    pass
