import numpy as np
import scipy.stats as sps
import statsmodels.api as sm
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

from dist_utils import prob_post, value_mh_cand, prob_mh_cand
from dist_utils import prob_HMC

# markov hastings sampler
def MH(x, Y, groups, var_mg, var_t, var_ss, T, verbose=True):
    xt_mh = []
    cands_mh = []
    status_mh = []
    for t in tqdm(range(T), disable=not verbose):
        # 1. propose candidate according to candidate distribution
        cand = torch.tensor(value_mh_cand(theta=x, var_mg=var_mg, var_t=var_t, var_ss=var_ss))

        # 2. Calculate acceptance probability
        post_ratio = (prob_post(theta=cand, Y=Y, groups=groups) / 
                      prob_post(theta=x, Y=Y, groups=groups)).detach().numpy()
        cand_ratio = (prob_mh_cand(x=x, theta=cand, var_mg=var_mg, var_t=var_t, var_ss=var_ss) / 
                      prob_mh_cand(x=cand, theta=x, var_mg=var_mg, var_t=var_t, var_ss=var_ss))
        prob_accept = np.minimum(1, post_ratio*cand_ratio)

        # 3. Accept or reject
        u = sps.uniform.rvs(loc=0, scale=1)
        if u <= prob_accept:
            cands_mh += [cand.detach().numpy()]
            status_mh += ['a']
            x = cand
            xt_mh += [cand.detach().numpy()]
        else:
            cands_mh += [cand.detach().numpy()]
            xt_mh += [x.detach().numpy()]
            status_mh += ['r']

    xt_mh = np.array(xt_mh)
    cands_mh = np.array(cands_mh)
    status_mh = np.array(status_mh)
    
    return xt_mh, cands_mh, status_mh


# leapfrog integrator
def leapfrog(q, p, Y, groups, M_inv, eps):
    q_cand = q.detach().requires_grad_()
    
    # a. update p
    log_prob_post = torch.log(prob_post(theta=q, Y=Y, groups=groups))
    log_prob_post.backward()
    p_cand = p + (eps/2)*q.grad

    # b. update q
    q_cand = q + eps*torch.matmul(M_inv, p_cand)
    q_cand = q_cand.detach().requires_grad_()

    # c. update p
    log_prob_post = torch.log(prob_post(theta=q_cand, Y=Y, groups=groups))
    log_prob_post.backward()
    p_cand = p_cand + (eps/2)*q_cand.grad
    
    return q_cand, p_cand


# hamiltonian monte carlo
def HMC(q, p, Y, groups, M, M_inv, eps, L, T, verbose=True):
    q_cand = q.detach().clone().requires_grad_()
    q_hmc = []
    cands_hmc = []
    status_hmc = []
    for t in tqdm(range(T), disable=not verbose):
        # 1. draw p        
        p_cand = torch.tensor(sps.multivariate_normal.rvs(mean=np.repeat(0,len(p)), cov=M)).float()

        # 2. evolve system
        for l in range(L):
            q_cand, p_cand = leapfrog(q_cand, p_cand, Y=Y, groups=groups, M_inv=M_inv, eps=eps)

        # 3. Metropolis hastings with proposal distributions
        cand_ratio = (prob_HMC(theta=q_cand, p=p_cand, Y=Y, groups=groups, M_inv=M_inv) / 
                      prob_HMC(theta=q, p=p, Y=Y, groups=groups, M_inv=M_inv)).detach()
        prob_accept = np.minimum(1, cand_ratio)

        # Accept or reject
        u = sps.uniform.rvs(loc=0, scale=1)
        if u <= prob_accept:
            q = q_cand
            p = -p_cand
            q_hmc += [q_cand.detach().numpy()]
            cands_hmc += [q_cand.detach().numpy()]
            status_hmc += ['a']
        else:
            q_hmc += [q.detach().numpy()]
            cands_hmc += [q_cand.detach().numpy()]
            status_hmc += ['r']

    q_hmc = np.array(q_hmc)
    cands_hmc = np.array(cands_hmc)
    status_hmc = np.array(status_hmc)
    
    return q_hmc, cands_hmc, status_hmc