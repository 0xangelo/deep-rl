"""
A collection of PyTorch utility functions and module subclasses
"""
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import grad
from torch.optim.lr_scheduler import _LRScheduler
from torch.distributions import constraints
from torch.distributions.transforms import Transform


_NP_TO_PT = {
    np.float64: torch.float64,
    np.float32: torch.float32,
    np.float16: torch.float16,
    np.int64: torch.int64,
    np.int32: torch.int32,
    np.int16: torch.int16,
    np.int8: torch.int8,
    np.uint8: torch.uint8,
}


def flat_grad(*args, **kwargs):
    return torch.cat([g.reshape((-1,)) for g in grad(*args, **kwargs)])


def grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == "inf":
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


def explained_variance_1d(ypred, y):
    assert y.dim() == 1 and ypred.dim() == 1
    vary = y.var().item()
    if np.isclose(vary, 0):
        if ypred.var().item() > 1e-8:
            return 0
        else:
            return 1
    return 1 - (y - ypred).var().item() / (vary + 1e-8)


# ==============================
# Schedulers
# ==============================


class LinearLR(_LRScheduler):
    def __init__(self, optimizer, total_num_epochs, last_epoch=-1):
        super().__init__(optimizer, last_epoch=last_epoch)
        self.total_num_epochs = float(total_num_epochs)

    def get_lr(self):
        return [
            base_lr - (base_lr * (self.last_epoch / self.total_num_epochs))
            for base_lr in self.base_lrs
        ]


# ==============================
# Modules
# ==============================


class ToFloat(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.scale = 1 / 250.0 if dtype is np.uint8 else 1

    def forward(self, x):
        return x.float() * self.scale


class Concat(ToFloat):
    def forward(self, *args):
        return super().forward(torch.cat(args, dim=-1))


class OneHot(nn.Module):
    def __init__(self, n_cat):
        super().__init__()
        self.n_cat = n_cat

    def forward(self, x):
        return torch.eye(self.n_cat)[x]


class Flatten(nn.Module):
    def __init__(self, flat_size):
        super().__init__()
        self.flat_size = flat_size

    def forward(self, x):
        return x.reshape(-1, self.flat_size)


class ExpandVector(nn.Module):
    def __init__(self, vector):
        super().__init__()
        self.vector = nn.Parameter(vector)

    def forward(self, x):
        return self.vector.expand(len(x), -1)


def update_polyak(from_module, to_module, polyak):
    for source, target in zip(from_module.parameters(), to_module.parameters()):
        target.data.mul_(polyak).add_(1 - polyak, source.data)


# ==============================
# Transforms
# ==============================


class TanhTransform(Transform):
    domain = constraints.real
    codomain = constraints.interval(-1, +1)
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return torch.tanh(x)

    def _inverse(self, y):
        to_log1 = 1 + y
        to_log2 = 1 - y
        to_log1[to_log1 == 0] += torch.finfo(y.dtype).eps
        to_log2[to_log2 == 0] += torch.finfo(y.dtype).eps
        return (torch.log(to_log1) - torch.log(to_log2)) / 2

    def log_abs_det_jacobian(self, x, y):
        to_log = 1 - y.pow(2)
        to_log[to_log == 0] += torch.finfo(y.dtype).eps
        return torch.log(to_log)
