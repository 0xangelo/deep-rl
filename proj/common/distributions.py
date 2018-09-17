import torch
import torch.nn as nn
import torch.distributions as dist


class Normal(dist.Normal):
    def __init__(self, flatparam):
        loc, scale = torch.chunk(flatparam, 2, dim=1)
        super().__init__(loc=loc, scale=scale)
        
    def flatparam(self):
        return torch.cat((self.loc, self.scale), dim=1)

    def likelihood_ratios(self, other, variables):
        return torch.exp(self.log_prob(variables) - other.log_prob(variables))


class Categorical(dist.Categorical):
    def __init__(self, flatparam):
        super().__init__(logits=flatparam)

    def flatparam(self):
        return self.logits

    def likelihood_ratios(self, other, variables):
        return torch.exp(self.log_prob(variables) - other.log_prob(variables))


def make_pdtype(ac_space):
    from gym import spaces
    if isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1
        return Normal
    elif isinstance(ac_space, spaces.Discrete):
        return Categorical
    else:
        raise NotImplementedError

