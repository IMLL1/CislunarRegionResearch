import numpy as np
import sklearn.mixture
import scipy as sp


# NOTE: states is N x 6
def alt_density_1d(
    states_rotating: list,
    times: list,
    period: float,
    earth_x: float,
    num_points: int = 1e7,
    alt_range=None,
):
    num_points = int(num_points)
    pos = states_rotating[:, :3] - np.array([earth_x, 0, 0])

    t = np.linspace(0, period, num_points)
    pos_interp = np.array([np.interp(t, times, pos[:, dim]) for dim in range(3)]).T
    radius_interp = np.linalg.vector_norm(pos_interp, axis=1)
    y, x = np.histogram(
        radius_interp,
        range=alt_range,
        bins=max([num_points // 1000, 100]),
        density=True,
    )

    r_cdf = x
    cdf = np.array([0, *np.cumsum(y)]) * (x[1]-x[0])

    x = np.mean([x[1:], x[:-1]], axis=0)
    p = y
    r = x

    return p, r, cdf, r_cdf


def eval_GMM_PDF(model: sklearn.mixture.GaussianMixture, x):
    means = model.means_.flatten()
    wts = model.weights_.flatten()
    covars = model.covariances_
    prob_densities = np.zeros_like(x)
    for n in range(len(means)):
        cluster_dist = sp.stats.norm(loc=means[n], scale=np.sqrt(covars[n]))
        prob_densities += wts[n] * cluster_dist.pdf(x).squeeze()
    return prob_densities


def eval_GMM_CDF(model: sklearn.mixture.GaussianMixture, x):
    means = model.means_.flatten()
    wts = model.weights_.flatten()
    covars = model.covariances_
    gmm_cdf = np.zeros_like(x)
    for n in range(len(means)):
        cluster_dist = sp.stats.norm(loc=means[n], scale=np.sqrt(covars[n]))
        gmm_cdf += wts[n] * cluster_dist.cdf(x).squeeze()
    return gmm_cdf


def integrate_cdf(cdf1: list, integrand: list):
    assert len(cdf1) == len(integrand) + 1
    i = np.arange(len(cdf1) - 1)
    # get int_i^{i+1}p(x)dx for each distribution
    int1 = cdf1[i + 1] - cdf1[i]
    prod = int1 * integrand
    return np.sum(prod)