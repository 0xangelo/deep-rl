import torch, torch.nn as nn, torch.distributions as dists


class DiagNormal(dists.Independent):
    def __init__(self, flatparam):
        loc, scale = torch.chunk(flatparam, 2, dim=1)
        base_distribution = dists.Normal(loc=loc, scale=scale)
        reinterpreted_batch_ndims = 1
        super().__init__(base_distribution, reinterpreted_batch_ndims)
        
    def flatparam(self):
        return torch.cat((self.base_dist.loc, self.base_dist.scale), dim=1)

    def likelihood_ratios(self, other, variables):
        return torch.exp(self.log_prob(variables) - other.log_prob(variables))


class Categorical(dists.Categorical):
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
        return DiagNormal
    elif isinstance(ac_space, spaces.Discrete):
        return Categorical
    else:
        raise NotImplementedError


@dists.kl.register_kl(DiagNormal, DiagNormal)
def _kl_diagnormal(dist1, dist2):
    dist1_vars = dist1.variance
    dist2_vars = dist2.variance

    return torch.sum(
        ((dist1.mean - dist2.mean).pow(2) + dist1_vars - dist2_vars) /
        (2 * dist2_vars + 1e-8) + torch.log(dist2.stddev / dist1.stddev),
        dim=-1
    )
