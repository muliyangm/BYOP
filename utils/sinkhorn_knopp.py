# some codes are adapted from the following sources:
# SwAV (https://github.com/facebookresearch/swav)
# UNO (https://github.com/DonkeyShot21/UNO)
# TRSSL (https://github.com/nayeemrizve/TRSSL)

import torch
# import numpy as np

def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor


class SinkhornKnopp_prior_new(torch.nn.Module):
    def __init__(self, num_iters=3, epsilon=0.05):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon

    @torch.no_grad()
    def iterate(self, Q, prior):
        Q = shoot_infs(Q)
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / Q.shape[1]

        order = torch.argsort(Q.sum(1)).detach()  # get class order from the model predictions

        r = prior * (Q.shape[1] / Q.shape[0])     # class prior
        r = r.cuda(non_blocking=True)

        r[order] = torch.sort(r)[0]   # reorder the prior
        r = torch.clamp(r, min=1)     # a trade-off between train and test performance
        r /= r.sum()

        for it in range(self.num_iters):
            u = torch.sum(Q, dim=1)
            u = r / u
            u = shoot_infs(u)
            Q *= u.unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    @torch.no_grad()
    def forward(self, logits, prior_):
        prior = prior_ / prior_.max()
        q = logits / self.epsilon
        M = torch.max(q)
        q -= M
        q = torch.exp(q).t()
        return self.iterate(q, prior)
