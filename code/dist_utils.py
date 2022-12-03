import numpy as np
import scipy.stats as sps
from tqdm import tqdm
import matplotlib.pyplot as plt


# posterior probability
def prob_post(theta, Y, groups):
    
    # unravel theta
    ss, tau, mu1, mu2, gam1, gam2 = theta
    mu = np.array((mu1, mu2))
    gam = np.array((gam1, gam2))
    
    #get counts
    __, ns = np.unique(groups, return_counts=True)
    n = ns.sum()
    
    # make mean vector
    mean_set = [mu, gam, (mu + gam)/2, tau*mu + (1-tau)*gam]
    means = []
    for n_i, mean_i in zip(ns, mean_set):
        for i in range(n_i):
            means += [mean_i]
    means = np.array(means)
    
    # sum of squares
    sum_squares = np.sum(np.square(Y - means))
    
    # calculate unscaled p
    p = (1/ss)**(n+1) * np.exp((-1/(2*ss)) * sum_squares)
    return p


# Metropolis hasting candidate value
def value_mh_cand(theta, var_n=0.001, var_c=0.1):
    ss, tau, mu1, mu2, gam1, gam2 = theta
    std_n = np.sqrt(var_n)
    std_c = np.sqrt(var_c)
    
    x = np.hstack([
        sps.norm.rvs(loc=np.sqrt(ss), scale=std_c)**2,
        sps.truncnorm.rvs(a=(0-tau)/std_n, b=(1-tau)/std_n, 
                            loc=tau, scale=std_n),
        sps.multivariate_normal.rvs(mean=np.array([mu1, mu2, gam1, gam2]), cov=var_n)])
    return x

# metropolis hasting candidate probability
def prob_mh_cand(x, theta, var_n=0.001, var_c=0.1):
    x_ss, x_tau, x_mu1, x_mu2, x_gam1, x_gam2 = x
    ss, tau, mu1, mu2, gam1, gam2 = theta
    std_n = np.sqrt(var_n)
    std_c = np.sqrt(var_c)
    
    p = (sps.norm.pdf(x=np.sqrt(x_ss), loc=np.sqrt(ss), scale=std_c) *
         sps.truncnorm.pdf(x=x_tau, a=(0-tau)/std_n, b=(1-tau)/std_n, loc=tau, scale=std_n) * 
         sps.multivariate_normal.pdf(x=np.array([x_mu1, x_mu2, x_gam1, x_gam2]), 
                                     mean=np.array([mu1, mu2, gam1, gam2]), cov=var_n))
        
    return p


# gibbs candidate sigma squared
def val_gibbs_ss(block_theta, Y, groups):
    # unravel theta
    ss, tau, mu, gam = block_theta

    #get counts
    __, ns = np.unique(groups, return_counts=True)
    n = ns.sum()

    # define parameters and draw
    sum_square = (np.sum(np.square(Y[groups == 1] - mu)) + np.sum(np.square(Y[groups == 2] - gam)) + 
                 np.sum(np.square(Y[groups == 3] - (mu + gam)/2)) + np.sum(np.square(Y[groups == 4] - tau*mu - (1-tau)*gam)))

    prop_ss = sps.invgamma.rvs(a=(n/2), scale=np.sqrt(sum_square/2))
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