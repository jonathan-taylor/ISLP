import numpy as np

from .tree import LeafNode, SplitNode, Tree
from .likelihood import marginal_loglikelihood

class ParticleTree(object):
    """
    Particle tree
    """

    def __init__(self,
                 tree,
                 alpha_split,
                 beta_split,
                 log_weight,
                 missing_data,
                 ssv,
                 available_predictors,
                 m,
                 sigma,
                 mu_prior_mean,
                 mu_prior_std):

        self.tree = tree.copy()  # keeps the tree that we care at the moment
        self.alpha_split = alpha_split
        self.beta_split = beta_split
        self.expansion_nodes = [0]
        self.log_weight = log_weight
        self.missing_data = missing_data
        self.used_variates = []
        self.ssv = ssv
        self.available_predictors = available_predictors
        self.m = m
        self.sigma = sigma
        self.mu_prior_std = mu_prior_std
        self.mu_prior_mean = mu_prior_mean
        
    def sample_tree_sequential(
            self,
            X,
            resid):

        tree_grew, left_node, right_node = False, None, None
        if self.expansion_nodes:
            index_leaf_node = self.expansion_nodes.pop(0)
            # Probability that this node will remain a leaf node
            depth = self.tree[index_leaf_node].depth
            prob_split = self.alpha_split / (1 + depth)**self.beta_split

            if np.random.random() < prob_split:
                (tree_grew,
                 index_selected_predictor,
                 left_node,
                 right_node) = grow_tree(
                     self.tree,
                     index_leaf_node,
                     self.ssv,
                     self.available_predictors,
                     X,
                     self.missing_data)
                if tree_grew:
                    new_indexes = self.tree.idx_leaf_nodes[-2:]
                    self.expansion_nodes.extend(new_indexes)
                    self.used_variates.append(index_selected_predictor)

        return tree_grew, left_node, right_node

    def increment_loglikelihood(self,
                                resid,
                                left_node,
                                right_node):

        # this could happen if a split value was the largest or smallest in a leaf
        
        if (len(right_node.idx_data_points) == 0 or
            len(left_node.idx_data_points) == 0):
            return 0

        right = marginal_loglikelihood(resid[right_node.idx_data_points],
                                       self.sigma,
                                       self.mu_prior_mean,
                                       self.mu_prior_std)

        left = marginal_loglikelihood(resid[left_node.idx_data_points],
                                      self.sigma,
                                      self.mu_prior_mean,
                                      self.mu_prior_std)

        full_idx = np.hstack([left_node.idx_data_points, right_node.idx_data_points])
        full = marginal_loglikelihood(resid[full_idx],
                                      self.sigma,
                                      self.mu_prior_mean,
                                      self.mu_prior_std)
        return left + right - full
        
    def marginal_loglikelihood(self,
                               resid):
        logL = 0
        for leaf_id in self.tree.idx_leaf_nodes:
            leaf_node = self.tree.get_node(leaf_id)
            if len(leaf_node.idx_data_points) > 0:
                logL += marginal_loglikelihood(resid[leaf_node.idx_data_points],
                                               self.sigma,
                                               self.mu_prior_mean,
                                               self.mu_prior_std)
        return logL

    def sample_values(self,
                      resid):
        
        for leaf_id in self.tree.idx_leaf_nodes:
            leaf_node = self.tree.get_node(leaf_id)
            if len(leaf_node.idx_data_points) > 0:
                nleaf = len(leaf_node.idx_data_points)

                quad = nleaf / self.sigma**2 + 1 / self.mu_prior_std**2
                linear = resid[leaf_node.idx_data_points].sum() / self.sigma**2 + self.mu_prior_mean / self.mu_prior_std**2

                mean = linear / quad
                std = 1. / np.sqrt(quad)
                leaf_node.value = np.random.normal() * std + mean
                #print(mean, std, resid[leaf_node.idx_data_points].mean(), quad * self.sigma**2, linear * self.sigma**2, self.sigma**2, 'sample')
            else:
                leaf_node.value = np.random.normal() * self.mu_prior_std + self.mu_prior_mean

# Section 2.5 of Lakshminarayanan

def grow_tree(
        tree,
        index_leaf_node,
        split_prior,
        available_predictors,
        X,
        missing_data):

    current_node = tree.get_node(index_leaf_node)
    idx_data_points = current_node.idx_data_points

    index_selected_predictor = split_prior.rvs()
    selected_predictor = available_predictors[index_selected_predictor]
    available_splitting_values = X[idx_data_points, selected_predictor]
    if missing_data:
        _keep = ~np.isnan(available_splitting_values)
        idx_data_points = idx_data_points[_keep]
        available_splitting_values = available_splitting_values[_keep]

    if available_splitting_values.size == 0:
        return False, None, None, None

    idx_selected_splitting_values = discrete_uniform_sampler(len(available_splitting_values))
    split_value = available_splitting_values[idx_selected_splitting_values]

    left_node_idx_data_points, right_node_idx_data_points = get_new_idx_data_points(
        split_value, idx_data_points, selected_predictor, X
    )

    new_split_node = SplitNode(
        index=index_leaf_node,
        idx_split_variable=selected_predictor,
        split_value=split_value,
    )

    new_left_node = LeafNode(
        index=current_node.get_idx_left_child(),
        value=np.nan,
        idx_data_points=left_node_idx_data_points,
    )
    new_right_node = LeafNode(
        index=current_node.get_idx_right_child(),
        value=np.nan,
        idx_data_points=right_node_idx_data_points,
    )

    tree.grow_tree(index_leaf_node, new_split_node, new_left_node, new_right_node)

    return True, index_selected_predictor, new_left_node, new_right_node


def get_new_idx_data_points(split_value, idx_data_points, selected_predictor, X):

    left_idx = X[idx_data_points, selected_predictor] <= split_value
    left_node_idx_data_points = idx_data_points[left_idx]
    right_node_idx_data_points = idx_data_points[~left_idx]

    return left_node_idx_data_points, right_node_idx_data_points


def discrete_uniform_sampler(upper_value):
    """Draw an integer from the uniform distribution with bounds [0, upper_value).

    This is the same and np.random.randint(upper_value) but faster.
    """
    return int(np.random.random() * upper_value)

__all__ = ['ParticleTree']
