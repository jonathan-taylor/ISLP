import numpy as np

class SampleSplittingVariable(object):

    def __init__(self, alpha_prior):
        """
        Sample splitting variables proportional to `alpha_prior`.

        This is equivalent as sampling weights from a Dirichlet distribution with `alpha_prior`
        parameter and then using those weights to sample from the available spliting variables.
        This enforce sparsity.
        """
        self.enu = list(enumerate(np.cumsum(alpha_prior / alpha_prior.sum())))

    def rvs(self):
        r = np.random.random()
        for i, v in self.enu:
            if r <= v:
                return i

def marginal_loglikelihood(resid,
                           sigma,
                           mu_std,
                           mu_mean):
    n = resid.shape[0]
    return 0 # - n / 2 * np.log(2 * np.pi * sigma**2) - 0.5 * np.sum((resid - mu)**2) / sigma**2

def compute_prior_probability(alpha):
    """
    Calculate the probability of the node being a LeafNode (1 - p(being SplitNode)).
    Taken from equation 19 in [Rockova2018].

    Parameters
    ----------
    alpha : float

    Returns
    -------
    list with probabilities for leaf nodes

    References
    ----------
    .. [Rockova2018] Veronika Rockova, Enakshi Saha (2018). On the theory of BART.
    arXiv, `link <https://arxiv.org/abs/1810.00787>`__
    """
    prior_leaf_prob = [0]
    depth = 1
    while prior_leaf_prob[-1] < 1:
        prior_leaf_prob.append(1 - alpha ** depth)
        depth += 1
    return prior_leaf_prob

