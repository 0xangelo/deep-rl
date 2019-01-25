# Inspired by OpenAI baselines:
# https://github.com/openai/baselines/blob/master/baselines/common/distributions.py
import torch, torch.nn as nn, torch.distributions as dists
from abc import ABC, abstractmethod


# ==============================
# Distribution types
# ==============================

class DistributionType(ABC, nn.Module):
    """ Maps flat vectors to probability distirbutions.

    Expects its input to be a tensor of size (N, in_features),
    where N is the batch dimension.
    """

    @property
    @abstractmethod
    def pd_class(self):
        pass

    @property
    @abstractmethod
    def param_shape(self):
        pass

    @property
    @abstractmethod
    def sample_shape(self):
        pass

    @abstractmethod
    def forward(self, feats):
        pass

    def from_flat(self, flat_params):
        return self.pd_class(flat_params)


class DiagNormalPDType(DistributionType):
    def __init__(self, size, in_features):
        super().__init__()
        self.size = size
        self.logstd = nn.Parameter(torch.zeros(size))
        self.mu = nn.Linear(in_features, size)
        nn.init.orthogonal_(self.mu.weight, gain=0.01)
        nn.init.constant_(self.mu.bias, 0)

    @property
    def pd_class(self):
        return DiagNormal

    @property
    def param_shape(self):
        return (self.size * 2,)

    @property
    def sample_shape(self):
        return (self.size,)

    def forward(self, feats):
        mu = self.mu(feats)
        stddev = self.logstd.expand_as(mu).exp()
        return self.from_flat(torch.cat((mu, stddev), dim=-1))


class CategoricalPDType(DistributionType):
    def __init__(self, n_cat, in_features):
        super().__init__()
        self.n_cat = n_cat
        self.logits = nn.Linear(in_features, n_cat)
        nn.init.orthogonal_(self.logits.weight, gain=0.01)
        nn.init.constant_(self.logits.bias, 0)

    @property
    def pd_class(self):
        return Categorical

    @property
    def param_shape(self):
        return (self.n_cat,)

    @property
    def sample_shape(self):
        return ()

    def forward(self, feats):
        return self.from_flat(self.logits(feats))


# ==============================
# Distributions
# ==============================

class Distribution(ABC, dists.Distribution):
    """ Probability distribution constructed from flat vectors.

    Extends torch.Distribution and replaces default constructor with
    a single argument version that expects a 2-D tensor (including
    batch dimension). Makes it easier to batch distribution parameters.
    """

    @property
    @abstractmethod
    def flat_params(self):
        pass

    def detach(self):
        return type(self)(self.flat_params.detach())


class DiagNormal(Distribution, dists.Independent):
    def __init__(self, flatparam):
        loc, scale = torch.chunk(flatparam, 2, dim=-1)
        base_distribution = dists.Normal(loc=loc, scale=scale)
        reinterpreted_batch_ndims = 1
        super().__init__(base_distribution, reinterpreted_batch_ndims)

    @property
    def flat_params(self):
        return torch.cat((self.base_dist.loc, self.base_dist.scale), dim=-1)


@dists.kl.register_kl(DiagNormal, DiagNormal)
def _kl_diagnormal(dist1, dist2):
    return torch.sum(
        ((dist1.mean - dist2.mean).pow(2) + dist1.variance - dist2.variance) /
        (2 * dist2.variance + 1e-8) + torch.log(dist2.stddev / dist1.stddev),
        dim=-1
    )


class Categorical(Distribution, dists.Categorical):
    def __init__(self, params):
        super().__init__(logits=params)

    @property
    def flat_params(self):
        return self.logits


def pdtype(ac_space, in_features):
    from gym import spaces
    if isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1
        return DiagNormalPDType(ac_space.shape[0], in_features)
    elif isinstance(ac_space, spaces.Discrete):
        return CategoricalPDType(ac_space.n, in_features)
    else:
        raise NotImplementedError
