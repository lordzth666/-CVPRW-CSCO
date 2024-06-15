import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import kendalltau
from scipy.stats import pearsonr
from tqdm import tqdm

class Plain_Loader:
    def __init__(self, ndims=5):
        self.ndims = ndims
        pass

    def make_features(self, org_features):
        """
        Make features using normalized graph laplacian, eigen-solver and maybe the Fiedler vector.
        :param org_features: (batch, num_stages, nodes, nodes)
        :return: (batch, num_stages*ndims)
        """
        num_examples = org_features.shape[0]
        num_stages = org_features.shape[1]
        n = org_features.shape[2]
        print("Building Plain Data Loader ...")

        new_features = org_features.copy()
        new_features = new_features.reshape([num_examples, -1])

        print(new_features.shape)
        return new_features

class RandomPathSampler:
    def __init__(self, map, min_length=1, max_length=5, forbidden_nodes=None):
        self.map = map
        self.min_length = min_length
        self.max_length = max_length
        self.nodes = np.shape(map)[0]
        self.forbidden_nodes = forbidden_nodes

        self.best_metric = -999
        self.best_path = []

    def sample_and_evaluate(self, beta=0.15):
        # Randomly pick up a start node.
        cur_node = 0
        path = []
        path_length = 0
        steps = 0

        dropout_prob = 1.0 / self.max_length

        while cur_node < self.nodes:
            num_choices = self.nodes - cur_node
            if np.random.rand() < dropout_prob:
                break
            if num_choices == 1:
                break
            else:
                choice = np.random.choice(num_choices-1) + 1
            next_node = cur_node + choice
            if next_node in self.forbidden_nodes:
                continue
            path_length += np.log(self.map[cur_node, next_node])
            path.append((cur_node, next_node))
            cur_node = next_node
            steps += 1
        if steps == 0:
            return -999, path
        else:
            return path_length + beta * np.log(steps), path

    def fiterout_n_paths(self, max_iters=1e5):
        for i in tqdm(range(int(max_iters))):
            success = False
            while not success:
                path_length, path = self.sample_and_evaluate(beta=0.02)
                if len(path) >= self.min_length and len(path) <= self.max_length:
                    success = True
                    if path_length > self.best_metric or \
                            (path_length == self.best_metric and len(path) > len(self.best_path)):
                        self.best_metric = path_length
                        self.best_path = path
            pass

def get_pearson_coefficient(X, y):
    # Get covariance first.
    # cov = np.mean((X-np.mean(X))*(y-np.mean(y)))
    # return cov / (np.std(X) * np.std(y))
    r, _ = pearsonr(X, y)
    return r

def get_kendalltau_coefficient(X, y):

    def get_rank(array):
        temp = array.argsort()
        ranks = np.arange(len(array))[temp.argsort()]
        return ranks

    r, _ = kendalltau(get_rank(X), get_rank(y))
    return r


def swap_nodes(mask, i, j):

    """
    Swap node i to node j in the adj matrix.
    """
    mask_ = mask.copy()
    mask_[i, :] = mask[j, :].copy()
    mask_[j, :] = mask[i, :].copy()
    mask_[:, i] = mask[:, j].copy()
    mask_[:, j] = mask[:, i].copy()
    return mask_

def find_largest_ops_def(node_ops_def_list):
    l_ksize = 0
    ret_index = -1
    for i in range(len(node_ops_def_list)):
        op = node_ops_def_list[i]
        if op['kernel_size'] > l_ksize:
            l_ksize = op['kernel_size']
            ret_index = i
    return ret_index

def generate_single_path_mask(best_mask, node_ops_def=None):

    def generate_critical_path(mask, node_ops_def):
        source = 0
        inf = 99999999
        n = np.shape(mask)[0]

        # Initialize
        dist = np.zeros(n)
        backtrace_ptr = np.zeros(n, dtype=np.int)-1
        backtrace_kernel_size = np.ones(n, dtype=np.int)
        for i in range(n):
            if mask[source][i] == 1:
                dist[i] = 1
                backtrace_ptr[i] = source
            else:
                dist[i] = -inf
        dist[0] = -inf
        
        for i in range(n):
            for j in range(i):
                if mask[j][i] == 1:
                    if dist[j] + 1 > dist[i] or (dist[j] + 1 == dist[i]
                    and node_ops_def[j]['kernel_size'] > backtrace_kernel_size[i]):
                        dist[i] = dist[j] + 1
                        backtrace_ptr[i] = j
                        backtrace_kernel_size[i] = node_ops_def[backtrace_ptr[i]]['kernel_size']

        # Find the longest path in the cell
        max_degree = np.max(dist)
        end_points_list = np.arange(n)[dist == max_degree]
        end_points = np.random.choice(end_points_list)

        cur_node = end_points
        node_list = []
        print(dist)
        print(backtrace_ptr)
        print(backtrace_kernel_size)
        while cur_node != 0:
            node_list.append(cur_node)
            cur_node = backtrace_ptr[cur_node]
        node_list.append(0)
        node_list = np.asarray(node_list)
        node_list = node_list[::-1]
        return node_list, dist

    def _create_single_path_mask(node_list, n):
        mask = np.zeros([n, n])
        for i in range(len(node_list)-1):
            mask[node_list[i], node_list[i+1]] = 1
        return mask

    depth = np.shape(best_mask)[0]
    width = np.shape(best_mask)[1]
    n = np.shape(best_mask)[2]
    
    node_coefficient = []
    final_mask = []
    
    for i in range(depth):
        for j in range(width):
            mask_ = best_mask[i][j]
            critical_path, dist = generate_critical_path(mask_, node_ops_def[i][j])
            # print(critical_path)
            n = np.shape(mask_)[0]
            # Now, find the node coefficient for each node.
            node_coefficient_ = np.ones(n)
            # Now, build the final mask.
            final_mask_ = _create_single_path_mask(critical_path, n)
            final_mask.append(final_mask_)
            idx = 0
            critical_path_ = critical_path.copy()
            path_length = len(critical_path_)
            for k_idx in range(path_length):
                coef_ = np.maximum(1, (dist == idx).sum())
                idx += 1
                k = critical_path_[k_idx]
                node_coefficient_[k] = coef_

            node_coefficient.append(node_coefficient_)
            critical_path = sorted(critical_path)
            print(critical_path)
            print(node_coefficient_[critical_path])
            
    # Finally, reshape node_coefficient for final use.
    node_coefficient = np.asarray(node_coefficient)
    node_coefficient = np.reshape(node_coefficient, [depth, width, n])
    # Finally, build the mask
    final_mask = np.reshape(final_mask, [depth, width, n, n])
    
    return final_mask, node_coefficient


def sat_max_edge_limit(mask, edge_limit=7):
    for i in range(mask.shape[0]):
        if np.sum(mask[i]) > edge_limit:
            return False
    return True


def sat_min_edge_limit(mask, edge_limit=0):
    for i in range(mask.shape[0]):
        if np.sum(mask[i]) < edge_limit:
            return False
    return True
