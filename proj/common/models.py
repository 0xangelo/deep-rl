import math
import numpy as np
import torch
import torch.nn as nn
from proj.common.utils import flatten_dim
from proj.common.distributions import *

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
        self.in_features = flatten_dim(ob_space)

        self.layers = nn.ModuleList()

class MlpModel(Model):
    def __init__(self, ob_space, ac_space, *, hidden_sizes=[64, 64], activation=nn.Tanh(), **kwargs):
        super().__init__(ob_space, ac_space, **kwargs)
        self.activation = activation
        in_features = self.in_features
        for out_features in hidden_sizes:
            layer = nn.Linear(in_features, out_features)
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
            nn.init.constant_(layer.bias, 0)
            self.layers.append(layer)
            self.layers.append(self.activation)
            in_features = out_features
        self.out_features = in_features

# ==============================
# Policies
# ==============================

class Policy(Model):
    def __init__(self, ob_space, ac_space, **kwargs):
        super().__init__(ob_space, ac_space, **kwargs)
        # Try to make action probabilities as close as possible
        out_features = flatten_dim(ac_space)
        layer = nn.Linear(self.out_features, out_features)
        nn.init.orthogonal_(layer.weight, gain=1)
        nn.init.constant_(layer.bias, 0)
        self.layers.append(layer)
        self.out_features = out_features
        
        self.distribution = make_pdtype(ac_space)
        
        if issubclass(self.distribution, Normal):
            self.logstd = nn.Parameter(torch.zeros(1, self.ac_dim))
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
        """
        Given some observations, returns the action distributions

        Arguments:
        x (Tensor): A batch of observations

        return (Tensor): distribution parameters
        """
        for module in self.layers:
            x = module(x)
        return self._features(x)

    @torch.no_grad()
    def get_actions(self, obs):
        features = self(torch.Tensor(obs))
        dists = self.distribution(features)
        actions = dists.sample()
        return actions, features

    def dists(self, obs):
        return self.distribution(self(obs))
        

class MlpPolicy(Policy, MlpModel):
    pass

# ==============================
# Baselines
# ==============================

class Baseline(Model):
    def __init__(self, ob_space, ac_space, *, mixture_fraction=0.1, **kwargs):
        super().__init__(ob_space, ac_space, **kwargs)
        self.mixture_fraction = mixture_fraction
        self.layers.append(nn.Linear(self.out_features, 1))
        self.out_features = 1

    def forward(self, x):
        """
        Given some observations, compute the baselines
        
        Arguments:
        param x (Tensor): A batch of observations

        return (Tensor): baselines for each observation
        """
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


class MlpBaseline(Baseline, MlpModel):
    pass
