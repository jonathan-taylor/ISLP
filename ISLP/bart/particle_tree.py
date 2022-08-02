




import numpy as np

from .tree import LeafNode, SplitNode, Tree
from .likelihood import marginal_loglikelihood, incremental_loglikelihood

class ParticleTree(object):
    """
    Particle tree
    """

    def __init__(self,
                 tree,
                 resid,
                 log_weight,
                 split_prob,
                 X_missing,
                 ssv,
                 available_predictors,
                 m,
                 sigmasq,
                 mu_prior_mean,
                 mu_prior_var,
                 random_state):

        self.tree = tree # keeps the tree that we care at the moment
        self.split_prob = split_prob
        self.expansion_nodes = [0]
        self.X_missing = X_missing
        self.used_variates = []
        self.ssv = ssv
        self.available_predictors = available_predictors
        self.m = m
        self.sigmasq = sigmasq
        self.mu_prior_var = mu_prior_var
        self.mu_prior_mean = mu_prior_mean
        self.random_state = random_state
        
        self.resid = resid
        self.log_weight = log_weight
        
    def sample_tree_sequential(self,
                               X,
                               X_quantiles,
                               resid):

        tree_grew, left_node, right_node = False, None, None
        if self.expansion_nodes:
            index_leaf_node = self.expansion_nodes.pop(0)
            # Probability that this node will remain a leaf node
            depth = self.tree[index_leaf_node].depth

            if self.random_state.random() < self.split_prob(depth):
                (tree_grew,
                 index_selected_predictor,
                 left_node,
                 right_node) = grow_tree(
                     self.tree,
                     index_leaf_node,
                     self.ssv,
                     self.available_predictors,
                     X,
                     X_quantiles,
                     self.X_missing,
                     self.random_state)
                if tree_grew:
                    new_indexes = self.tree.idx_leaf_nodes[-2:]
                    self.expansion_nodes.extend(new_indexes)
                    self.used_variates.append(index_selected_predictor)
                 
        return tree_grew, left_node, right_node

    def increment_loglikelihood(self,
                                left_node,
                                right_node):

        # this could happen if a split value was the largest or smallest in a leaf
        
        # if (np.sum(right_node.idx_data_points) == 0 or
        #     np.sum(left_node.idx_data_points) == 0):
        #     return 0

        if (len(right_node.idx_data_points) == 0 or
            len(left_node.idx_data_points) == 0):
            return 0

        (logL_increment,
         left_moments,
         right_moments) =  incremental_loglikelihood(self.resid,
                                                     left_node.idx_data_points,
                                                     right_node.idx_data_points,
                                                     self.sigmasq,
                                                     self.mu_prior_mean,
                                                     self.mu_prior_var)
        left_node._response_moments = left_moments
        right_node._response_moments = right_moments
        return logL_increment

    def marginal_loglikelihood(self):

        logL = 0
        for leaf_id in self.tree.idx_leaf_nodes:
            leaf_node = self.tree.get_node(leaf_id)
            if len(leaf_node.idx_data_points) > 0:
#            if np.sum(leaf_node.idx_data_points) > 0:
                response_moments = None
                if hasattr(leaf_node, "_response_moments"):
                    response_moments = leaf_node._response_moments
                leaf_logL, response_moments =  marginal_loglikelihood(None, 
                                                                      self.sigmasq,
                                                                      self.mu_prior_mean,
                                                                      self.mu_prior_var,
                                                                      response_moments=response_moments,
                                                                      incremental=True) # we can use this to avoid computing sum(response**2)
                logL += leaf_logL
                leaf_node._response_moments = response_moments
                
        return logL

    def sample_values(self,
                      resid):
        
        for leaf_id in self.tree.idx_leaf_nodes:
            leaf_node = self.tree.get_node(leaf_id)
            if len(leaf_node.idx_data_points) > 0:
            # if np.sum(leaf_node.idx_data_points) > 0:
            #     nleaf = np.sum(leaf_node.idx_data_points) 
                if hasattr(leaf_node, "_response_moments"):
                    nleaf, resid_sum = leaf_node._response_moments[:2]
                else:
                    resid_sum = resid[leaf_node.idx_data_points].sum()
                    nleaf = len(leaf_node.idx_data_points)

                quad = nleaf / self.sigmasq + 1 / self.mu_prior_var
                linear = resid_sum / self.sigmasq + self.mu_prior_mean / self.mu_prior_var

                mean = linear / quad
                std = 1. / np.sqrt(quad)
                leaf_node.value = self.random_state.normal() * std + mean
            else:
                leaf_node.value = self.random_state.normal() * np.sqrt(self.mu_prior_var) + self.mu_prior_mean

    def set_resid(self, resid):
        self.resid = resid
        
        # set the sum of resid correctly in each leaf
        for leaf_id in self.tree.idx_leaf_nodes:
            leaf_node = self.tree.get_node(leaf_id)
            resid_idx = self.resid[leaf_node.idx_data_points]
            leaf_node._response_moments = (resid_idx.shape[0],
                                           resid_idx.sum(),
                                           0)

# Section 2.5 of Lakshminarayanan

def discrete_uniform_sampler(upper_value,
                             random_state):
    """Draw an integer from the uniform distribution with bounds [0, upper_value).
    This is the same and np.random.randit(upper_value) but faster.
    """
    return int(random_state.random() * upper_value)


def grow_tree(
        tree,
        index_leaf_node,
        split_prior,
        available_predictors,
        X,
        X_quantiles,
        X_missing,
        random_state):

    current_node = tree.get_node(index_leaf_node)
    idx_data_points = current_node.idx_data_points

    index_selected_predictor = split_prior.rvs()
    selected_predictor = available_predictors[index_selected_predictor]
    X_select = X[:, selected_predictor]
    if X_quantiles is not None:
        available_splitting_values = X_quantiles[:, selected_predictor]
    else:
        available_splitting_values = X_select[idx_data_points]
        if X_missing:
            _keep = ~np.isnan(available_splitting_values)
            idx_data_points = idx_data_points[_keep]
            available_splitting_values = available_splitting_values[_keep]

    if available_splitting_values.size == 0:
        return False, None, None, None

    split_idx = discrete_uniform_sampler(available_splitting_values.shape[0],
                                         random_state)
    split_value = available_splitting_values[split_idx]

    left_node_idx_data_points, right_node_idx_data_points = get_new_idx_data_points(
        split_value, idx_data_points, X_select)

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


def get_new_idx_data_points(split_value, idx_data_points, X_select):

    left_idx = X_select[idx_data_points] <= split_value
    left_node_idx_data_points = idx_data_points[left_idx]
    right_node_idx_data_points = idx_data_points[~left_idx]

    return left_node_idx_data_points, right_node_idx_data_points

__all__ = ['ParticleTree']
