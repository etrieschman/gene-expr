import numpy as np
import scipy.stats as sps
import statsmodels.api as sm
from tqdm import tqdm
import matplotlib.pyplot as plt


# plot settings
def set_plt_settings():
    plt.rcParams.update({'font.size': 18})
    SMALL_SIZE = 12
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 20

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# plot 2D distributions
def plot_dist(cands, plot_status=False, status=np.nan, alpha=0.2):
    fig, ax = plt.subplots(ncols=3, figsize=(16, 6))

    if not plot_status:
        ax[0].scatter(cands[:, 0], cands[:, 1], color='blue', label='accept', alpha=alpha)
        ax[1].scatter(cands[:, 2], cands[:, 3], color='blue', label='accept', alpha=alpha)
        ax[2].scatter(cands[:, 4], cands[:, 5], color='blue', label='accept', alpha=alpha)
    else:
        # plot rejections
        ix = np.where(status == 'r')
        ax[0].scatter(cands[ix, 0], cands[ix, 1], color='red', label='reject', alpha=alpha)
        ax[1].scatter(cands[ix, 2], cands[ix, 3], color='red', label='reject', alpha=alpha)
        ax[2].scatter(cands[ix, 4], cands[ix, 5], color='red', label='reject', alpha=alpha)
        # plot acceptances
        ix = np.where(status == 'a')
        ax[0].scatter(cands[ix, 0], cands[ix, 1], color='blue', label='accept', alpha=alpha)
        ax[1].scatter(cands[ix, 2], cands[ix, 3], color='blue', label='accept', alpha=alpha)
        ax[2].scatter(cands[ix, 4], cands[ix, 5], color='blue', label='accept', alpha=alpha)
        # plot legend and title
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        n_accept = (status == 'a').sum()
        ax[1].set(title=f'n accept: {n_accept} ({n_accept / len(status):0.2%})')

    ax[0].set(xlabel='ss', ylabel='tau')
    ax[1].set(xlabel='mu1', ylabel='mu2')
    ax[2].set(xlabel='gam1', ylabel='gam2')
    plt.show()

# plot marginal distributions
def plot_marginals(xt, alpha=0.5):
    theta_labels = ['ss', 'tau', 'mu1', 'mu2', 'gam1', 'gam2']
    fig, ax = plt.subplots(ncols=2, figsize=(16, 6), sharey=True, width_ratios=[3,1])

    for i in range(xt.shape[1]):
        ax[0].plot(xt[:,i], alpha=alpha, label=theta_labels[i])
        ax[1].hist(xt[:,i], alpha=alpha, density=True, bins=40, orientation='horizontal', label=theta_labels[i])

    ax[1].legend()
    ax[0].set(xlabel='iteration', title='samples')
    ax[1].set(title='marginal densities')
    plt.show()

# plot autocorrelation
def plot_acorr(xt, nlags=5000):
    theta_labels = ['ss', 'tau', 'mu1', 'mu2', 'gam1', 'gam2']

    plt.subplots(figsize=(12, 6))

    for i in range(xt.shape[1]):
        plt.plot(sm.tsa.acf(xt[:,i], nlags=nlags), alpha=0.75, label=theta_labels[i])

    plt.legend()
    plt.xlabel('lag')
    plt.title('autocorrelation')
    plt.show()