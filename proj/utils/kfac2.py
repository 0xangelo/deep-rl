"""
MIT License

Copyright (c) 2018 Thomas George, César Laurent and Université de Montréal.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

Adapted from: https://github.com/Thrandis/EKFAC-pytorch
"""
import contextlib, torch
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp


def inv_worker(conn, use_pi, eps):
    """Inverses the covariances."""
    pi = 1.0
    try:
        while True:
            cmd, mats = conn.recv()
            if cmd == "close":
                break
            for xxt, ggt, ixxt, iggt, num_locations in mats:
                # Computes pi
                if use_pi:
                    tx = torch.trace(xxt) * ggt.shape[0]
                    tg = torch.trace(ggt) * xxt.shape[0]
                    pi = (tx / tg)
                # Regularizes and inverse
                eps = eps / num_locations
                diag_xxt = xxt.new(xxt.shape[0]).fill_((eps * pi) ** 0.5)
                diag_ggt = ggt.new(ggt.shape[0]).fill_((eps / pi) ** 0.5)
                ixxt.copy_((xxt + torch.diag(diag_xxt)).inverse())
                iggt.copy_((ggt + torch.diag(diag_ggt)).inverse())
            conn.send(mats)
    except KeyboardInterrupt:
        print('inv_worker: got KeyboardInterrupt')
    finally:
        conn.close()


class KFACOptimizer(optim.Optimizer):
    def __init__(self, net, eps, sua=False, pi=False, update_freq=1,
                 alpha=1.0, kl_clip=1e-3, eta=1.0):
        """ K-FAC Optimizer for Linear and Conv2d layers.

        Computes the K-FAC of the second moment of the gradients.
        It works for Linear and Conv2d layers and silently skip other layers.

        Args:
            net (torch.nn.Module): Network to optimize.
            eps (float): Tikhonov regularization parameter for the inverses.
            sua (bool): Applies SUA approximation.
            pi (bool): Computes pi correction for Tikhonov regularization.
            update_freq (int): Perform inverses every update_freq updates.
            alpha (float): Running average parameter (if == 1, no r. ave.).
            kl_clip (float): Scale the gradients by the squared fisher norm.
            eta (float): upper bound for gradient scaling.
        """
        self.eps = eps
        self.sua = sua
        self.pi = pi
        self.update_freq = update_freq
        self.alpha = alpha
        self.kl_clip = kl_clip
        self.eta = eta
        self._recording = False
        self.params = []
        self._iteration_counter = 0

        param_set = set()
        for mod in net.modules():
            mod_class = type(mod).__name__
            if mod_class in ['Linear', 'Conv2d']:
                mod.register_forward_pre_hook(self._save_input)
                mod.register_backward_hook(self._save_grad_output)
                info = (mod.kernel_size, mod.padding, mod.stride) \
                       if mod_class == 'Conv2d' else None
                params = [mod.weight]
                if mod.bias is not None:
                    params.append(mod.bias)
                d = {'params': params, 'info': info, 'layer_type': mod_class}
                self.params.append(d)
                param_set.update(set(params))

        param_list = [p for p in net.parameters() if p not in param_set]
        self.params.append({'params': param_list})
        super(KFACOptimizer, self).__init__(self.params, {})

        self._nprocs = min(len(self.param_groups[:-1]), mp.cpu_count())
        self.workers = []
        self.conns = []
        for _ in range(self._nprocs):
            worker_conn, master_conn = mp.Pipe()
            worker = mp.Process(
                target=inv_worker, args=(worker_conn, self.pi, self.eps))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
            self.conns.append(master_conn)

    def step(self):
        """Preconditions and applies gradients."""
        fisher_norm = 0.
        states = []
        for group in self.param_groups[:-1]:
            # Update convariances and inverses
            state = self.state[group['params'][0]]
            self._compute_covs(group, state)
            if self._iteration_counter == 0:
                state['ixxt'] = torch.zeros_like(state['xxt'])
                state['iggt'] = torch.zeros_like(state['ggt'])
            states.append(tuple(state[k] for k in [
                'xxt', 'ggt', 'ixxt', 'iggt', 'num_locations']))

        if self._iteration_counter % self.update_freq == 0:
            for idx, conn in enumerate(self.conns):
                beg = (len(states)//self._nprocs) * idx
                end = len(states) if idx == self._nprocs-1 else (
                    beg + len(states)//self._nprocs)
                conn.send((None, states[beg:end]))
            results = []
            for conn in self.conns:
                results.extend(conn.recv())
            for group, mats in zip(self.param_groups[:-1], results):
                weight = group['params'][0]
                self.state[weight]['xxt'] = mats[0]
                self.state[weight]['ggt'] = mats[1]
                self.state[weight]['ixxt'] = mats[2]
                self.state[weight]['iggt'] = mats[3]

        for group in self.param_groups[:-1]:
            # Getting parameters
            params = group['params']
            weight, bias = params if len(params) == 2 else (params[0], None)
            state = self.state[weight]

            # Preconditionning
            gw, gb = self._precond(weight, bias, group, state)
            # Updating gradients
            fisher_norm += (weight.grad * gw).sum()
            weight.grad.data = gw
            if bias is not None:
                fisher_norm += (bias.grad * gb).sum()
                bias.grad.data = gb

        fisher_norm += sum(
            (p.grad * p.grad).sum() for p in self.param_groups[-1]['params'])

        # Eventually scale the norm of the gradients and apply each
        scale = min(self.eta, torch.sqrt(self.kl_clip / fisher_norm))
        for group in self.param_groups:
            for param in group['params']:
                param.grad.data *= scale
                param.data.sub_(param.grad.data)
        self._iteration_counter += 1

    @contextlib.contextmanager
    def record_stats(self):
        try:
            self._recording = True
            yield
        except Exception as excep:
            raise excep
        finally:
            self._recording = False

    def _save_input(self, mod, i):
        """Saves input of layer to compute covariance."""
        if self._recording:
            self.state[mod.weight]['x'] = i[0]
            # self.state[mod.weight]['x'] = torch.cat(
            #     (self.state[mod.weight].get('x', torch.tensor([])), i[0]))

    def _save_grad_output(self, mod, grad_input, grad_output):
        """Saves grad on output of layer to compute covariance."""
        if self._recording:
            self.state[mod.weight]['gy'] = grad_output[0]*grad_output[0].size(0)

    def _precond(self, weight, bias, group, state):
        """Applies preconditioning."""
        if group['layer_type'] == 'Conv2d' and self.sua:
            return self._precond_sua(weight, bias, group, state)
        ixxt = state['ixxt']
        iggt = state['iggt']
        g = weight.grad.data
        s = g.shape
        if group['layer_type'] == 'Conv2d':
            g = g.contiguous().view(s[0], s[1]*s[2]*s[3])
        if bias is not None:
            gb = bias.grad.data
            g = torch.cat([g, gb.view(gb.shape[0], 1)], dim=1)
        g = torch.mm(torch.mm(iggt, g), ixxt)
        if group['layer_type'] == 'Conv2d':
            g /= state['num_locations']
        if bias is not None:
            gb = g[:, -1].contiguous().view(*bias.shape)
            g = g[:, :-1]
        else:
            gb = None
        g = g.contiguous().view(*s)
        return g, gb

    def _precond_sua(self, weight, bias, group, state):
        """Preconditioning for KFAC SUA."""
        ixxt = state['ixxt']
        iggt = state['iggt']
        g = weight.grad.data
        s = g.shape
        g = g.permute(1, 0, 2, 3).contiguous()
        if bias is not None:
            gb = bias.grad.view(1, -1, 1, 1).expand(1, -1, s[2], s[3])
            g = torch.cat([g, gb], dim=0)
        g = torch.mm(ixxt, g.contiguous().view(-1, s[0]*s[2]*s[3]))
        g = g.view(-1, s[0], s[2], s[3]).permute(1, 0, 2, 3).contiguous()
        g = torch.mm(iggt, g.view(s[0], -1)).view(s[0], -1, s[2], s[3])
        g /= state['num_locations']
        if bias is not None:
            gb = g[:, -1, s[2]//2, s[3]//2]
            g = g[:, :-1]
        else:
            gb = None
        return g, gb

    def _compute_covs(self, group, state):
        """Computes the covariances."""
        weight = group['params'][0]
        x, gy = state.pop('x'), state.pop('gy')
        # Computation of xxt
        if group['layer_type'] == 'Conv2d':
            if not self.sua:
                kernel_size, padding, stride = group['info']
                x = F.unfold(x, kernel_size, padding=padding, stride=stride)
            else:
                x = x.view(x.shape[0], x.shape[1], -1)
            x = x.data.permute(1, 0, 2).contiguous().view(x.shape[1], -1)
        else:
            x = x.data.t()
        if len(group['params']) == 2:
            ones = torch.ones_like(x[:1])
            x = torch.cat([x, ones], dim=0)
        if self._iteration_counter == 0:
            state['xxt'] = torch.mm(x, x.t()) / float(x.shape[1])
        else:
            state['xxt'].addmm_(mat1=x, mat2=x.t(),
                                beta=(1. - self.alpha),
                                alpha=self.alpha / float(x.shape[1]))
        # Computation of ggt
        if group['layer_type'] == 'Conv2d':
            gy = gy.data.permute(1, 0, 2, 3)
            state['num_locations'] = gy.shape[2] * gy.shape[3]
            gy = gy.contiguous().view(gy.shape[0], -1)
        else:
            gy = gy.data.t()
            state['num_locations'] = 1
        if self._iteration_counter == 0:
            state['ggt'] = torch.mm(gy, gy.t()) / float(gy.shape[1])
        else:
            state['ggt'].addmm_(mat1=gy, mat2=gy.t(),
                                beta=(1. - self.alpha),
                                alpha=self.alpha / float(gy.shape[1]))

    def _inv_covs(self, xxt, ggt, num_locations):
        """Inverses the covariances."""
        # Computes pi
        pi = 1.0
        if self.pi:
            tx = torch.trace(xxt) * ggt.shape[0]
            tg = torch.trace(ggt) * xxt.shape[0]
            pi = (tx / tg)
        # Regularizes and inverse
        eps = self.eps / num_locations
        diag_xxt = xxt.new(xxt.shape[0]).fill_((eps * pi) ** 0.5)
        diag_ggt = ggt.new(ggt.shape[0]).fill_((eps / pi) ** 0.5)
        ixxt = (xxt + torch.diag(diag_xxt)).inverse()
        iggt = (ggt + torch.diag(diag_ggt)).inverse()
        return ixxt, iggt

    def __del__(self):
        for conn in self.conns:
            conn.close()
        for worker in self.workers:
            worker.join()
