# Inspired by OpenAI baselines:
# https://github.com/openai/baselines/blob/master/baselines/common/distributions.py
import torch
import torch.nn as nn
import torch.distributions as dists
from abc import ABC, abstractmethod
from proj.utils.torch_util import ExpandVector, TanhTransform, AffineTransform

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

    @abstractmethod
    def from_flat(self, flat_params):
        pass


LOG_STD_MAX = 2
LOG_STD_MIN = -20

class DiagNormalPDType(DistributionType):
    def __init__(self, size, in_features, *, indep_std=True):
        super().__init__()
        self.size = size
        self.mu = nn.Linear(in_features, size)
        nn.init.orthogonal_(self.mu.weight, gain=0.01)
        nn.init.constant_(self.mu.bias, 0)
        if indep_std:
            self.logstd = ExpandVector(torch.zeros(size))
        else:
            self.logstd = nn.Linear(in_features, size)

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
        stddev = torch.clamp(self.logstd(feats), LOG_STD_MIN, LOG_STD_MAX).exp()
        return self.from_flat(torch.cat((mu, stddev), dim=-1))

    def from_flat(self, flat_params):
        return self.pd_class(flat_params)


class ClampedDiagNormalPDType(DiagNormalPDType):
    def __init__(self, size, in_features, *, indep_std=True, low, high):
        super().__init__(size, in_features, indep_std=indep_std)
        self.low = low
        self.high = high

    @property
    def pd_class(self):
        return ClampedDiagNormal

    def from_flat(self, flat_params):
        return ClampedDiagNormal(flat_params, self.low, self.high)


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

    def from_flat(self, flat_params):
        return self.pd_class(flat_params)

# ==============================
# Distributions
# ==============================

class Distribution(ABC):
    """ Probability distribution constructed from flat vectors.

    Extends torch.Distribution and replaces default constructor with
    a single argument version that expects a 2-D tensor (including
    batch dimension). Makes it easier to batch distribution parameters.
    """

    @property
    @abstractmethod
    def mode(self):
        pass

    @property
    @abstractmethod
    def flat_params(self):
        pass

    @abstractmethod
    def detach(self):
        pass


class DiagNormal(dists.Independent, Distribution):
    def __init__(self, flatparam):
        loc, scale = torch.chunk(flatparam, 2, dim=-1)
        base_distribution = dists.Normal(loc=loc, scale=scale)
        reinterpreted_batch_ndims = 1
        super().__init__(base_distribution, reinterpreted_batch_ndims)

    @property
    def mode(self):
        return self.mean

    @property
    def flat_params(self):
        return torch.cat((self.base_dist.loc, self.base_dist.scale), dim=-1)

    def detach(self):
        return DiagNormal(self.flat_params.detach())


@dists.kl.register_kl(DiagNormal, DiagNormal)
def _kl_diagnormal(dist1, dist2):
    return torch.sum(
        ((dist1.mean - dist2.mean).pow(2) + dist1.variance - dist2.variance) /
        (2 * dist2.variance + 1e-8) + torch.log(dist2.stddev / dist1.stddev),
        dim=-1
    )


class ClampedDiagNormal(dists.TransformedDistribution, Distribution):
    def __init__(self, flatparam, low, high):
        base_distribution = DiagNormal(flatparam)
        self.loc = (high+low) / 2
        self.scale = (high-low) / 2
        super().__init__(
            base_distribution,
            [
                TanhTransform(cache_size=1),
                AffineTransform(self.loc, self.scale, cache_size=1, event_dim=1)
            ]
        )

    @property
    def mean(self):
        mu = self.base_dist.mean
        for transform in self.transforms:
            mu = transform(mu)
        return mu

    @property
    def mode(self):
        return self.mean

    @property
    def flat_params(self):
        return self.base_dist.flat_params

    def detach(self):
        return ClampedDiagNormal(self.flat_params.detach(),
                                 self.loc - self.scale, self.loc + self.scale)


class Categorical(dists.Categorical, Distribution):
    def __init__(self, params):
        super().__init__(logits=params)

    @property
    def mode(self):
        self.logits.argmax(1)

    @property
    def flat_params(self):
        return self.logits

    def detach(self):
        return Categorical(self.logits.detach())


def pdtype(ac_space, in_features, *, clamp_acts=False, indep_std=True):
    from gym import spaces
    if isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1
        if clamp_acts:
            return ClampedDiagNormalPDType(
                ac_space.shape[0], in_features, indep_std=indep_std,
                low=torch.Tensor(ac_space.low),
                high=torch.Tensor(ac_space.high))
        else:
            return DiagNormalPDType(
                ac_space.shape[0], in_features, indep_std=indep_std)
    elif isinstance(ac_space, spaces.Discrete):
        return CategoricalPDType(ac_space.n, in_features)
    else:
        raise NotImplementedError
