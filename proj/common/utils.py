import gym, torch, random, numpy as np
from torch.autograd import grad
from torch.distributions.kl import kl_divergence as kl
from .distributions import Normal, Categorical

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


def compute_flat_grad(output, inputs, filter_input_ids=set(), retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True

    inputs = list(inputs)
    params = []
    for i, param in enumerate(inputs):
        if i not in filter_input_ids:
            params.append(param)

    grads = grad(output, params, retain_graph=retain_graph, create_graph=create_graph)

    j = 0
    out_grads = []
    for i, param in enumerate(inputs):
        if i in filter_input_ids:
            out_grads.append(torch.zeros(param.numel()))
        else:
            out_grads.append(grads[j].view(-1))
            j += 1
    grads = torch.cat(out_grads)

    for param in params:
        param.grad = None
    return grads


def get_mu_fim(policy, obs):
    if issubclass(policy.pdtype, Normal):
        mean = policy.dists(obs).mean
        ####
        cov_inv = policy.logstd.exp().pow(-2).squeeze(0).repeat(len(obs))
        param_count = 0
        std_index = 0
        Id = 0
        for name, param in self.named_parameters():
            if name == "logstd":
                std_id = Id
                std_index = param_count
            param_count += param.numel()
            Id += 1
        return cov_inv.detach(), mean, {'std_id': std_id, 'std_index': std_index}
    else:
        probs = policy.dists(obs).probs
        M = probs.pow(-1).view(-1)
        return M.detach(), probs, {}


def fisher_vector_product(v, obs, policy, damping=1e-3):
    M, mu, info = get_mu_fim(policy, obs)
    mu = mu.view(-1)
    filter_input_ids = set() if issubclass(policy.pdtype, Categorical) else set([info['std_id']])

    t = torch.ones(mu.size(), requires_grad=True)
    mu_t = (mu * t).sum()
    Jt = compute_flat_grad(
        mu_t,
        policy.parameters(), filter_input_ids=filter_input_ids, create_graph=True)
    Jtv = (Jt * v).sum()
    Jv = grad(Jtv, t)[0]
    MJv = M * Jv.detach()
    mu_MJv = (MJv * mu).sum()
    JTMJv = compute_flat_grad(mu_MJv, policy.parameters(), filter_input_ids=filter_input_ids).detach()
    JTMJv /= len(obs)
    if issubclass(policy.pdtype, Normal):
        std_index = info['std_index']
        JTMJv[std_index: std_index + len(M)] += 2 * v[std_index: std_index + len(M)]
    return JTMJv + v * damping
    

# def fisher_vector_product(v, obs, policy, damping=1e-3):
#     dists = policy.dists(obs)
#     avg_kl = kl(policy.pdtype(dists.flatparam().detach()), dists).mean()
#     grads = grad(avg_kl, policy.parameters(), create_graph=True)
#     flat_grads = torch.cat([grad.view(-1) for grad in grads])
#     fvp = grad((flat_grads * v).sum(), policy.parameters())
#     flat_fvp = torch.cat([g.contiguous().view(-1).data for g in fvp])
#     return flat_fvp + v * damping


def explained_variance_1d(ypred, y):
    assert y.dim() == 1 and ypred.dim() == 1
    vary = y.var().item()
    if np.isclose(vary, 0):
        if ypred.var().item() > 1e-8:
            return 0
        else:
            return 1
    return 1 - torch.var(y - ypred).item() / (vary + 1e-8)


