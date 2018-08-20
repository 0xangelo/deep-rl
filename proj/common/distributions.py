import torch
import torch.nn as nn
import torch.distributions as dist


class Normal(dist.Normal):
    def flatparam(self):
        return torch.cat((self.loc, self.scale), dim=1).numpy()

    @classmethod
    def fromflat(cls, flatparam):
        loc, scale = torch.chunk(flatparam, 2, dim=1)
        return cls(loc=loc, scale=scale)

class Categorical(dist.Categorical):
    def flatparam(self):
        return self.logits.numpy()

    @classmethod
    def fromflat(cls, flatparam):
        return cls(logits=flatparam)


def make_pdtype(ac_space):
    from gym import spaces
    if isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1
        return Normal
    elif isinstance(ac_space, spaces.Discrete):
        return Categorical
    else:
        raise NotImplementedError

