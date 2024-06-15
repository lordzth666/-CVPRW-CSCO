import numpy as np
from math import sqrt

def random_subsampling(n):
    """
    Random sampling weights according to mapping
    :param n:
    :param map:
    :return:
    """
    num_nodes_to_sample = np.random.randint(low=1, high=n)
    nodes_collection = np.random.choice(np.arange(n), num_nodes_to_sample, replace=False)
    # Starting point 0 must be added
    if not 0 in nodes_collection:
        np.append(nodes_collection, 0)
    nodes_queue = [0]
    weight_adj_matrix = np.zeros([n, n])
    branching_factor = np.random.uniform(3. / 4, 4. / 3, size=None)
    for i in range(n):
        if not i in nodes_collection:
            continue
        nodes_avail = len(nodes_collection[nodes_collection<i])
        thresh = branching_factor  / (nodes_avail + 1)
        for j in range(i):
            if not j in nodes_queue:
                continue
            p = np.random.rand()
            if p < thresh:
                weight_adj_matrix[j][i] = 1.
                if not i in nodes_queue:
                    nodes_queue.append(i)
                if not j in nodes_queue:
                    nodes_queue.append(j)
    nodes_queue.sort()
    return weight_adj_matrix, nodes_queue
    pass


def random_connected_graph(n,
                     min_edges=2,
                     max_edges=10,
                     max_leafs=99999,
                     min_non_leafs=2):

    while 1:
        node_list = [0]
        adjacency_matrix = np.zeros([n, n], dtype=np.float32)
        num_edges = np.random.randint(min_edges, max_edges+1)
        for i in range(num_edges):
            # Select a node from a nodelist.
            while 1:
                node = np.random.choice(node_list)
                max_offset = n - node
                if max_offset <= 1:
                    continue
                offset = np.random.randint(1, max_offset)
                if adjacency_matrix[node, node+offset] == 0:
                    break
            if not node + offset in node_list:
                node_list.append(node+offset)
            adjacency_matrix[node, node+offset] = 1

        # Finally, discard any inception-like structures. These structures are commonly sampled.
        out_degree = np.sum(adjacency_matrix, axis=1)
        in_degree = np.sum(adjacency_matrix, axis=1)
        avail_non_leafs = (out_degree != 0).sum()
        avail_leafs = (out_degree == 0).sum()
        if avail_non_leafs >= 2 and np.max(in_degree) <= 3:
            break

    return adjacency_matrix, node_list

def random_connected_graph_single_leaf(n,
                     min_edges=2,
                     max_edges=10):

    node_list = [0]
    adjacency_matrix = np.zeros([n, n], dtype=np.float32)

    num_edges = np.random.randint(min_edges, max_edges+1)

    for i in range(num_edges):
            # Select a node from a nodelist.
            while 1:
                node = np.random.choice(node_list)
                max_offset = n - node
                if max_offset <= 1:
                    continue
                offset = np.random.randint(1, max_offset)
                if adjacency_matrix[node, node+offset] == 0:
                    break
            if not node + offset in node_list:
                node_list.append(node+offset)
            adjacency_matrix[node, node+offset] = 1

    # Finally, collect all leafs and concatenate it to the largest node.
    largest_node_idx = np.max(node_list)
    out_degree = np.sum(adjacency_matrix, axis=1)
    for node_idx in node_list:
        if out_degree[node_idx] == 0 and node_idx != largest_node_idx:
            adjacency_matrix[node_idx, largest_node_idx] = 1

    return adjacency_matrix, node_list


def random_single_path(n,
                       min_edges=2,
                       max_edges=10,
                       max_leafs=99999,
                       min_non_leafs=2):
    node_list = [0]
    adjacency_matrix = np.zeros([n, n], dtype=np.float32)

    num_edges = np.random.randint(min_edges, max_edges + 1)

    for i in range(num_edges):
        node = node_list[-1]
        max_offset = n - node
        if max_offset <= 1:
            break
        offset = np.random.randint(1, max_offset)
        node_list.append(node + offset)
        adjacency_matrix[node, node + offset] = 1

    return adjacency_matrix, node_list


def adaptive_subsampling(n, weights_prior=None):
    """
    :param n:
    :return:
    """
    assert weights_prior is not None
    nodes_queue = [0]
    weight_adj_matrix = np.zeros([n, n])
    for i in range(n):
        if not i in nodes_queue:
            continue
        for j in range(i+1, n):
            p = np.random.rand()
            if p < weights_prior[i][j]:
                weight_adj_matrix[i][j] = 1.
                if not i in nodes_queue:
                    nodes_queue.append(i)
                if not j in nodes_queue:
                    nodes_queue.append(j)
    nodes_queue.sort()
    return weight_adj_matrix, nodes_queue


def fetch_degree_matrix(masks):
    """
    Fetch the degree matrix.
    :param masks:
    :return: (in_degree_matrix, out_degree_matrix)
    """
    n = np.shape(masks)[0]
    in_degree_matrix = np.zeros(n)
    out_degree_matrix = np.zeros(n)
    # Starting from node 0.
    in_degree_matrix[0] = 1.
    for i in range(n):
        for j in range(i+1, n):
            if masks[i][j] == 1:
                in_degree_matrix[j] += 1
                out_degree_matrix[i] += 1
    return in_degree_matrix, out_degree_matrix


def sumsampling_and_fetch_degree(n, weights_prior=None, policy='random'):
    """
    :param n:
    :param weights_prior:
    :param policy:
    :return:
    """
    if policy == "random":
        masks, nodes_queue = random_subsampling(n)
    elif policy == 'adaptive':
        assert weights_prior is not None
        masks, nodes_queue = adaptive_subsampling(n, weights_prior=weights_prior)
    else:
        raise NotImplementedError
    in_degree_matrix, out_degree_matrix = fetch_degree_matrix(masks)
    return masks, nodes_queue, in_degree_matrix, out_degree_matrix


def fetch_upper_triangle(weight_matrix):
    """
    Fetch the upper trianglular part of the weight matrix and flatten
    :param weight_matrix:
    :return:
    """
    weight_matrix = np.asarray(weight_matrix)
    if len(np.shape(weight_matrix)) == 2:
        n = np.shape(weight_matrix)[0]
    elif len(np.shape(weight_matrix)) == 3:
        n = np.shape(weight_matrix)[1]
    else:
        raise NotImplementedError("Weight matrix should be 2-D or 3-D!")

    ret_indices = np.triu_indices(n, k=1)
    if len(np.shape(weight_matrix)) == 2:
        return weight_matrix[ret_indices]
    else:
        weight_matrix = np.transpose(weight_matrix, (1, 2, 0))
        ret_values = weight_matrix[ret_indices]
        return np.transpose(ret_values, (1, 0))

def recover_from_upper_triangle(weight_matrix, nodes=30, depth=3, width=3):
    sparse_weight_matrix = np.zeros([depth, width, nodes, nodes])
    for d1 in range(depth):
        for d2 in range(depth):
            idx = 0
            for i in range(nodes):
                for j in range(i + 1, nodes):
                    sparse_weight_matrix[d1][d2][i][j] = weight_matrix[d1][d2][idx]
                    idx += 1
            assert idx == int(nodes * (nodes - 1) / 2)
    return sparse_weight_matrix





