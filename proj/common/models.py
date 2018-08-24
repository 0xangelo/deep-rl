import numpy as np
import torch
import torch.nn as nn
from proj.common.utils import flatten_dim
from proj.common.input import observation_input
from proj.common.distributions import *

torch.set_default_tensor_type(torch.FloatTensor)


# ==============================
# Models
# ==============================

class Model(nn.Module):
    def __init__(self, ob_space, ac_space, **kwargs):
        super().__init__()
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.process_obs = observation_input(ob_space)
        self.ob_dim = flatten_dim(ob_space)
        self.ac_dim = flatten_dim(ac_space)

        self.layers = nn.ModuleList()

class MlpModel(Model):
    def __init__(self, ob_space, ac_space, *, hidden_sizes=[64, 64], activation=nn.Tanh(), **kwargs):
        super().__init__(ob_space, ac_space, **kwargs)
        self.activation = activation
        in_features = self.ob_dim
        for out_features in hidden_sizes:
            self.layers.append(nn.Linear(in_features, out_features))
            self.layers.append(self.activation)
            in_features = out_features
        self.layers.append(nn.Linear(in_features, self.ac_dim))

# ==============================
# Policies
# ==============================

class Policy(Model):
    def __init__(self, ob_space, ac_space, **kwargs):
        super().__init__(ob_space, ac_space, **kwargs)
        # Try to make action probabilities as close as possible
        nn.init.orthogonal_(self.layers[-1].weight, gain=1)
        self.layers[-1].bias.data.mul_(0)
        self.distribution = make_pdtype(ac_space)
        
        if issubclass(self.distribution, Normal):
            self.logstd = nn.Parameter(torch.zeros(1, self.ac_dim))
            def make_dist(layers_out):
                logstd = self.logstd.expand_as(layers_out)
                sttdev = torch.exp(logstd)
                return self.distribution(loc=layers_out, scale=sttdev)
        else:
            def make_dist(layers_out):
                return self.distribution(logits=layers_out)
        self._make_dist = make_dist

    def forward(self, x):
        """
        Given some observations, returns the action distributions

        Arguments:
        x (Tensor): A batch of observations

        return (Distribution): Actions distributions
        """
        for module in self.layers:
            x = module(x)
        return self._make_dist(x)

    @torch.no_grad()
    def get_actions(self, obs):
        dists = self(self.process_obs(obs))
        actions = dists.sample()
        return actions.numpy(), dists.flatparam()
        
    def get_action(self, ob):
        actions, dists = self.get_actions(ob[np.newaxis])
        return actions[0], dists[0]


class MlpPolicy(Policy, MlpModel):
    pass

# ==============================
# Baselines
# ==============================

class Baseline(Model):
    def __init__(self, ob_space, ac_space, *, mixture_fraction=0.1, **kwargs):
        super().__init__(ob_space, ac_space, **kwargs)
        self.mixture_fraction = mixture_fraction
        self.layers.append(nn.Linear(self.ac_dim, 1))

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

    @torch.no_grad()
    def predict(self, obs):
        return self(self.process_obs(obs)).numpy()

    def update(self, trajs):
        obs = np.concatenate([traj['observations'] for traj in trajs])
        returns = np.concatenate([traj['returns'] for traj in trajs])
        baselines = np.concatenate([traj['baselines'] for traj in trajs])

        targets = self.mixture_fraction * baselines + \
                  (1 - self.mixture_fraction) * returns

        obs = self.process_obs(obs)
        targets = torch.as_tensor(targets)

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
