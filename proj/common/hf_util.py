import torch
from torch.distributions.kl import kl_divergence
from proj.utils.torch_util import flat_grad

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
