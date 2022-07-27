import numpy as np

from .tree import LeafNode, SplitNode, Tree
from .utils import marginal_loglikelihood

class ParticleTree(object):
    """
    Particle tree
    """

    def __init__(self,
                 tree,
                 log_weight,
                 likelihood,
                 missing_data,
                 ssv,
                 prior_prob_leaf_node,
                 available_predictors,
                 m,
                 sigma,
                 mu_mean,
                 mu_std):

        self.tree = tree.copy()  # keeps the tree that we care at the moment
        self.expansion_nodes = [0]
        self.log_weight = log_weight
        self.old_likelihood_logp = likelihood
        self.missing_data = missing_data
        self.used_variates = []
        self.ssv = ssv
        self.prior_prob_leaf_node = prior_prob_leaf_node
        self.available_predictors = available_predictors
        self.m = m
        self.sigma = sigma
        self.mu_std = mu_std
        self.mu_mean = mu_mean

    def sample_tree_sequential(
        self,
        X,
        Y,            
        sum_trees_output,
    ):
        tree_grew = False
        if self.expansion_nodes:
            index_leaf_node = self.expansion_nodes.pop(0)
            # Probability that this node will remain a leaf node
            prob_leaf = self.prior_prob_leaf_node[self.tree[index_leaf_node].depth]

            if prob_leaf < np.random.random():
                tree_grew, index_selected_predictor = grow_tree(
                    self.tree,
                    index_leaf_node,
                    self.ssv,
                    self.available_predictors,
                    X,
                    self.missing_data,
                    sum_trees_output,
                    self.m,
                    self.mu_std,
                    'constant'
                )
                if tree_grew:
                    new_indexes = self.tree.idx_leaf_nodes[-2:]
                    self.expansion_nodes.extend(new_indexes)
                    self.used_variates.append(index_selected_predictor)

        return tree_grew

    def update_weight(self,
                      Y: np.ndarray,
                      sum_trees_output_noi : np.ndarray) -> None:
        """
        Update the weight of a particle

        Since the prior is used as the proposal,the weights are updated additively as the ratio of
        the new and old log-likelihoods.
        """
        new_likelihood = marginal_loglikelihood(Y - sum_trees_output_noi,
                                                self.sigma,
                                                self.mu_std,
                                                self.mu_mean)
        self.log_weight += new_likelihood - self.old_likelihood_logp
        self.old_likelihood_logp = new_likelihood


def grow_tree(
    tree,
    index_leaf_node,
    ssv,
    available_predictors,
    X,
    missing_data,
    sum_trees_output,
    m,
    mu_std,
    response,
):
    mean = np.mean
    
    current_node = tree.get_node(index_leaf_node)
    idx_data_points = current_node.idx_data_points

    index_selected_predictor = ssv.rvs()
    selected_predictor = available_predictors[index_selected_predictor]
    available_splitting_values = X[idx_data_points, selected_predictor]
    if missing_data:
        idx_data_points = idx_data_points[~np.isnan(available_splitting_values)]
        available_splitting_values = available_splitting_values[
            ~np.isnan(available_splitting_values)
        ]

    if available_splitting_values.size == 0:
        return False, None

    idx_selected_splitting_values = discrete_uniform_sampler(len(available_splitting_values))
    split_value = available_splitting_values[idx_selected_splitting_values]
    new_split_node = SplitNode(
        index=index_leaf_node,
        idx_split_variable=selected_predictor,
        split_value=split_value,
    )

    left_node_idx_data_points, right_node_idx_data_points = get_new_idx_data_points(
        split_value, idx_data_points, selected_predictor, X
    )

    left_node_value = draw_leaf_value(
        sum_trees_output[left_node_idx_data_points],
        X[left_node_idx_data_points, selected_predictor],
        mean,
        m,
        mu_std,
        response,
    )
    right_node_value = draw_leaf_value(
        sum_trees_output[right_node_idx_data_points],
        X[right_node_idx_data_points, selected_predictor],
        mean,
        m,
        mu_std,
        response,
    )

    new_left_node = LeafNode(
        index=current_node.get_idx_left_child(),
        value=left_node_value,
        idx_data_points=left_node_idx_data_points,
    )
    new_right_node = LeafNode(
        index=current_node.get_idx_right_child(),
        value=right_node_value,
        idx_data_points=right_node_idx_data_points,
    )
    tree.grow_tree(index_leaf_node, new_split_node, new_left_node, new_right_node)

    return True, index_selected_predictor


def get_new_idx_data_points(split_value, idx_data_points, selected_predictor, X):

    left_idx = X[idx_data_points, selected_predictor] <= split_value
    left_node_idx_data_points = idx_data_points[left_idx]
    right_node_idx_data_points = idx_data_points[~left_idx]

    return left_node_idx_data_points, right_node_idx_data_points


def draw_leaf_value(Y_mu_pred, X_mu, mean, m, mu_std, response):
    """Draw Gaussian distributed leaf values"""

    if Y_mu_pred.size == 0:
        return 0
    else:
        norm = np.random.normal() * mu_std
        if Y_mu_pred.size == 1:
            mu_mean = Y_mu_pred.item() / m
        elif response == "constant":
            mu_mean = mean(Y_mu_pred) / m
        draw = norm + mu_mean
        return draw

def discrete_uniform_sampler(upper_value):
    """Draw an integer from the uniform distribution with bounds [0, upper_value).

    This is the same and np.random.randint(upper_value) but faster.
    """
    return int(np.random.random() * upper_value)

__all__ = ['ParticleTree']
