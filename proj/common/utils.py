import gym
import torch
from torch.autograd import grad
import numpy as np
import random

def set_global_seeds(seed):
    if seed is None:
        seed = random.randint(0,2**32)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return seed


def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    """
    Demmel p 312. Approximately solve x = A^{-1}b, or Ax = b, 
    where we only have access to f: x -> Ax
    """
    p = b.clone()
    r = b.clone()
    x = torch.zeros_like(b)
    rdotr = torch.dot(r,r)

    for i in range(cg_iters):
        z = f_Ax(p)
        v = rdotr / torch.dot(p,z)
        x += v * p
        r -= v * z
        newrdotr = torch.dot(r,r)
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    return x


def fisher_vector_product(v, avg_kl, policy, damping=1e-3):
    grads = grad(avg_kl(), policy.parameters(), create_graph=True)
    flat_grads = torch.cat([grad.view(-1) for grad in grads])
    fvp = grad((flat_grads * v).sum(), policy.parameters())
    flat_fvp = torch.cat([g.contiguous().view(-1).data for g in fvp])
    return flat_fvp + v * damping


def explained_variance_1d(ypred, y):
    assert y.dim() == 1 and ypred.dim() == 1
    vary = y.var().item()
    if np.isclose(vary, 0):
        if ypred.var().item() > 1e-8:
            return 0
        else:
            return 1
    return 1 - torch.var(y - ypred).item() / (vary + 1e-8)


