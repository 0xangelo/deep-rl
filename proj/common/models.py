import torch, torch.nn as nn, numpy as np
from abc import ABC, abstractmethod
from . import distributions as dists
from .observations import n_features

torch.set_default_tensor_type(torch.FloatTensor)
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

# ==============================
# Models
# ==============================

class Model(nn.Module):
    def __init__(self, ob_space, ac_space, **kwargs):
        super().__init__()
        self.ob_space = ob_space
        self.in_features = n_features(ob_space)

        self.layers = nn.ModuleList()

class MlpModel(Model):
    def __init__(self, ob_space, ac_space, *, hidden_sizes=[64, 64],
                 activation=nn.Tanh(), **kwargs):
        super().__init__(ob_space, ac_space, **kwargs)
        self.activation = activation
        in_features = self.in_features
        for out_features in hidden_sizes:
            layer = nn.Linear(in_features, out_features)
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0)
            self.layers.append(layer)
            self.layers.append(self.activation)
            in_features = out_features
        self.out_features = in_features

# ==============================
# Policies
# ==============================

class AbstractPolicy(ABC, Model):
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


class FeedForwardPolicy(AbstractPolicy):
    def __init__(self, ob_space, ac_space, **kwargs):
        super().__init__(ob_space, ac_space, **kwargs)
        # Try to make action probabilities as close as possible
        out_features = n_features(ac_space)
        layer = nn.Linear(self.out_features, out_features)
        nn.init.orthogonal_(layer.weight, gain=1)
        nn.init.constant_(layer.bias, 0)
        self.layers.append(layer)
        self.out_features = out_features
        
        self.pdtype = dists.make_pdtype(ac_space)
        
        if issubclass(self.pdtype, dists.Normal):
            self.logstd = nn.Parameter(torch.zeros(1, self.out_features))
            self.out_features *= 2
            def features(layers_out):
                logstd = self.logstd.expand_as(layers_out)
                sttdev = torch.exp(logstd)
                return torch.cat((layers_out, sttdev), dim=1)
        else:
            def features(layers_out):
                return layers_out
        self._features = features

    def forward(self, x):
        for module in self.layers:
            x = module(x)
        return self._features(x)

    @torch.no_grad()
    def actions(self, obs):
        return self.dists(obs).sample()

    def action(self, ob):
        return self.actions(ob.unsqueeze(0))[0]

    def dists(self, obs):
        return self.pdtype(self(obs))
        

class MlpPolicy(FeedForwardPolicy, MlpModel):
    pass

# ==============================
# Baselines
# ==============================

class AbstractBaseline(ABC, Model):
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
    def update(self, trajs):
        """
        Given a list of trajectories, update the reinforcement baseline.

        Args:
        trajs (list): A list of trajectories as returned from 
                            parallel_collect_samples
        """
        pass


class ZeroBaseline(AbstractBaseline):
    def forward(self, x):
        return torch.zeros(len(x))

    def update(self, trajs):
        pass


class FeedForwardBaseline(AbstractBaseline):
    def __init__(self, ob_space, ac_space, *, mixture_fraction=0.1, **kwargs):
        super().__init__(ob_space, ac_space, **kwargs)
        self.mixture_fraction = mixture_fraction
        self.layers.append(nn.Linear(self.out_features, 1))
        self.out_features = 1

    def forward(self, x):
        for module in self.layers:
            x = module(x)
        return torch.squeeze(x)

    def update(self, trajs):
        obs = torch.cat([traj['observations'] for traj in trajs])
        returns = torch.cat([traj['returns'] for traj in trajs])
        baselines = torch.cat([traj['baselines'] for traj in trajs])

        targets = self.mixture_fraction * baselines + \
                  (1 - self.mixture_fraction) * returns

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.LBFGS(self.parameters(), max_iter=10)
        def closure():
            optimizer.zero_grad()
            loss = loss_fn(self(obs), targets)
            loss.backward()
            return loss

        optimizer.step(closure)


class MlpBaseline(FeedForwardBaseline, MlpModel):
    pass
