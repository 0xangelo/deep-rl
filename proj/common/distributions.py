import torch, torch.nn as nn, torch.distributions as dists


class DiagNormal(dists.MultivariateNormal):
    def likelihood_ratios(self, other, variables):
        return torch.exp(self.log_prob(variables) - other.log_prob(variables))

    def kl_self(self):
        return dists.kl.kl_divergence(
            DiagNormal(
                self.loc.detach(),
                scale_tril=self.scale_tril.detach()
            ),
            self
        )

class Categorical(dists.Categorical):
    def likelihood_ratios(self, other, variables):
        return torch.exp(self.log_prob(variables) - other.log_prob(variables))

    def kl_self(self):
        return dists.kl.kl_divergence(
            dists.Categorical(logits=self.logits.detach()),
            self
        )


def pd_maker(ac_space, policy):
    from gym import spaces
    if isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1
        def build_pd(means):
            scale_tril = torch.diag(torch.exp(policy.logstd))
            return DiagNormal(means, scale_tril=scale_tril)
    elif isinstance(ac_space, spaces.Discrete):
        def build_pd(logits):
            return Categorical(logits=logits)
    else:
        raise NotImplementedError
    return build_pd


