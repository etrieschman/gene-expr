import numpy as np
import torch
import scipy.stats as sps
from tqdm import tqdm
import matplotlib.pyplot as plt

# posterior probability
def prob_post(theta, Y, groups, scale=1000):
    # unravel theta
    ss, tau, mu, gam = theta[0], theta[1], theta[2:4], theta[4:6]
    
    #get counts
    __, ns = np.unique(groups, return_counts=True)
    n = ns.sum()
    
    # sumsquare
    sum_square = (torch.sum(torch.square(Y[groups == 1] - mu)) + 
                  torch.sum(torch.square(Y[groups == 2] - gam)) + 
                  torch.sum(torch.square(Y[groups == 3] - (mu + gam)/2)) + 
                  torch.sum(torch.square(Y[groups == 4] - tau*mu - (1-tau)*gam)))
    
    # calculate unscaled p
    p = (1/ss)**(n+1) * torch.exp((-1/(2*ss)) * sum_square)
    # add boundary conditions for tau and sigma squared
    p = (p * 
        (1 / (1 + torch.exp(-scale*(tau)))) * 
        (1 / (1 + torch.exp(-scale*(-tau+1)))) * 
        (1 / (1 + torch.exp(-scale*(ss)))))
    return p


# Metropolis hasting candidate value
def value_mh_cand(theta, var_mg=0.001, var_t=0.1, var_ss=0.1):
    ss, tau, mu1, mu2, gam1, gam2 = theta
    std_mg = np.sqrt(var_mg)
    std_t = np.sqrt(var_t)
    std_ss = np.sqrt(var_ss)
    
    x = np.hstack([
        np.exp(sps.norm.rvs(loc=np.log(ss), scale=std_ss)),
        sps.truncnorm.rvs(a=(0-tau)/std_t, b=(1-tau)/std_t, 
                            loc=tau, scale=std_t),
        sps.multivariate_normal.rvs(mean=np.array([mu1, mu2, gam1, gam2]), cov=var_mg)])
    return x

# metropolis hasting candidate probability
def prob_mh_cand(x, theta, var_mg=0.001, var_t=0.1, var_ss=0.1):
    x_ss, x_tau, x_mu1, x_mu2, x_gam1, x_gam2 = x.detach().numpy()
    ss, tau, mu1, mu2, gam1, gam2 = theta.detach().numpy()
    std_mg = np.sqrt(var_mg)
    std_t = np.sqrt(var_t)
    std_ss = np.sqrt(var_ss)
    
    p = (sps.norm.pdf(x=np.log(x_ss), loc=np.log(ss), scale=std_ss) * 
         np.abs(np.exp(np.log(ss))) * # jacobian for transformation
         sps.truncnorm.pdf(x=x_tau, a=(0-tau)/std_t, b=(1-tau)/std_t, loc=tau, scale=std_t) * 
         sps.multivariate_normal.pdf(x=np.array([x_mu1, x_mu2, x_gam1, x_gam2]), 
                                     mean=np.array([mu1, mu2, gam1, gam2]), cov=var_mg))
        
    return p


# gibbs candidate sigma squared
def val_gibbs_ss(block_theta, Y, groups):
    # unravel theta
    ss, tau, mu, gam = block_theta

    #get counts
    __, ns = np.unique(groups, return_counts=True)
    n = ns.sum()

    # define parameters and draw
    sum_square = (np.sum(np.square(Y[groups == 1] - mu)) + 
                  np.sum(np.square(Y[groups == 2] - gam)) + 
                  np.sum(np.square(Y[groups == 3] - (mu + gam)/2)) + 
                  np.sum(np.square(Y[groups == 4] - tau*mu - (1-tau)*gam)))

    prop_ss = sps.invgamma.rvs(a=n, scale=sum_square/2)
    return prop_ss

# gibbs candidate tau
def val_gibbs_tau(block_theta, Y, groups, eps=0.00001):
    # unravel theta
    ss, __, mu, gam = block_theta

    #get counts
    __, ns = np.unique(groups, return_counts=True)
    n = ns.sum()

    # error handling in case mu == gam (divide by zero error)
    if (mu == gam).all():
        mu_mgam = eps
    else:
        mu_mgam = np.sum(np.square(mu - gam))
    
    mean_tau = np.dot((mu - gam), (Y[groups == 4] - gam).sum(axis=0)) / (ns[3] * mu_mgam)
    var_tau = ss / (ns[3] * mu_mgam)
    std_tau = np.sqrt(var_tau)
    tau = sps.truncnorm.rvs(a=(0 - mean_tau) / std_tau, 
                            b=(1 - mean_tau) / std_tau, 
                            loc=mean_tau, scale=std_tau)
    return tau

# gibbs candidate mu
def val_gibbs_mu(block_theta, Y, groups):
    # unravel theta
    ss, tau, __, gam = block_theta

    #get counts
    __, ns = np.unique(groups, return_counts=True)
    n = ns.sum()
    
    # calculate mean/var
    mu_denom = ns[0] + ns[2]/4 + ns[3]*(tau**2)
    mean_mu_num = (Y[groups == 1].sum(axis=0) + 0.5*Y[groups == 3].sum(axis=0) + 
                   tau*Y[groups == 4].sum(axis=0) - (ns[2]/4 + ns[3]*tau*(1-tau))*gam)
    mean_mu = mean_mu_num / mu_denom
    var_mu = ss / mu_denom
    std_mu = np.sqrt(var_mu)

    # generate sample
    prop_mu = np.array([sps.norm.rvs(loc=mean_mu[0], scale=std_mu),
                        sps.norm.rvs(loc=mean_mu[1], scale=std_mu)])
    return prop_mu


# gibbs candidate gamma
def val_gibbs_gam(block_theta, Y, groups):
    # unravel theta
    ss, tau, mu, __ = block_theta

    #get counts
    __, ns = np.unique(groups, return_counts=True)
    n = ns.sum()
    
    # calculate mean/var
    gam_denom = ns[1] + ns[2]/4 + ns[3]*((1-tau)**2)
    mean_gam_num = (Y[groups == 2].sum(axis=0) + 0.5*Y[groups == 3].sum(axis=0) + 
                    (1-tau)*Y[groups == 4].sum(axis=0) - 
                    (ns[2]/4 + ns[3]*tau*(1-tau))*mu)
    mean_gam = mean_gam_num / gam_denom
    var_gam = ss / gam_denom
    std_gam = np.sqrt(var_gam)

    # generate sample
    prop_gam = np.array([sps.norm.rvs(loc=mean_gam[0], scale=std_gam),
                        sps.norm.rvs(loc=mean_gam[1], scale=std_gam)])
    return prop_gam


# HMC function
def prob_HMC(theta, p, M_inv, Y, groups):
    prob = prob_post(theta, Y, groups)*torch.exp(-(1/2)*torch.matmul(p, torch.matmul(M_inv, p)))
    return prob