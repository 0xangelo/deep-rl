import torch, random, numpy as np
import scipy.signal
from torch.autograd import grad
from torch.distributions.kl import kl_divergence


def set_global_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def flat_grad(*args, **kwargs):
    return torch.cat([g.reshape((-1,)) for g in grad(*args, **kwargs)])


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
# Hessian-free optimization util
# ==============================

def fisher_vec_prod(v, obs, policy, damping=1e-3):
    dists = policy(obs)
    avg_kl = kl_divergence(dists.detach(), dists).mean()
    grad = flat_grad(avg_kl, policy.parameters(), create_graph=True)
    fvp = flat_grad(grad.dot(v), policy.parameters()).detach()
    return fvp + v * damping


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


@torch.no_grad()
def line_search(f, x0, dx, expected_improvement, y0=None, accept_ratio=0.1,
                backtrack_ratio=0.8, max_backtracks=15, atol=1e-7):
    if y0 is None:
        y0 = f(x0)

    if expected_improvement >= atol:
        for exp in range(max_backtracks):
            ratio = backtrack_ratio ** exp
            x = x0 - ratio * dx
            y = f(x)
            improvement = y0 - y
            # Armijo condition
            if improvement / (expected_improvement * ratio) >= accept_ratio:
                return x, expected_improvement * ratio, improvement

    return x0, expected_improvement, 0
