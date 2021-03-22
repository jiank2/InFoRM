import time

import numpy as np
import networkx as nx
import sklearn.preprocessing as skpp

from copy import deepcopy
from scipy.sparse import diags, isspmatrix_coo, triu


def trace(mat):
    """
    calculate trace of a sparse matrix
    :param mat: scipy.sparse matrix (csc, csr or coo)
    :return: Tr(mat)
    """
    return mat.diagonal().sum()


def row_normalize(mat):
    """
    normalize a matrix by row
    :param mat: scipy.sparse matrix (csc, csr or coo)
    :return: row-normalized matrix
    """
    degrees = np.asarray(mat.sum(axis=1).flatten())
    degrees = np.divide(1, degrees, out=np.zeros_like(degrees), where=degrees != 0)
    degrees = diags(np.asarray(degrees)[0,:])
    return degrees @ mat


def column_normalize(mat):
    """
    normalize a matrix by column
    :param mat: scipy.sparse matrix (csc, csr or coo)
    :return: column-normalized matrix
    """
    degrees = np.asarray(mat.sum(axis=0).flatten())
    degrees = np.divide(1, degrees, out=np.zeros_like(degrees), where=degrees != 0)
    degrees = diags(np.asarray(degrees)[0, :])
    return mat @ degrees


def symmetric_normalize(mat):
    """
    symmetrically normalize a matrix
    :param mat: scipy.sparse matrix (csc, csr or coo)
    :return: symmetrically normalized matrix
    """
    degrees = np.asarray(mat.sum(axis=0).flatten())
    degrees = np.divide(1, degrees, out=np.zeros_like(degrees), where=degrees != 0)
    degrees = diags(np.asarray(degrees)[0, :])
    degrees.data = np.sqrt(degrees.data)
    return degrees @ mat @ degrees


def jaccard_similarity(mat):
    """
    get jaccard similarity matrix
    :param mat: scipy.sparse.csc_matrix
    :return: similarity matrix of nodes
    """
    # make it a binary matrix
    mat_bin = mat.copy()
    mat_bin.data[:] = 1

    col_sum = mat_bin.getnnz(axis=0)
    ab = mat_bin.dot(mat_bin.T)
    aa = np.repeat(col_sum, ab.getnnz(axis=0))
    bb = col_sum[ab.indices]
    sim = ab.copy()
    sim.data /= (aa + bb - ab.data)
    return sim


def cosine_similarity(mat):
    """
    get cosine similarity matrix
    :param mat: scipy.sparse.csc_matrix
    :return: similarity matrix of nodes
    """
    mat_row_norm = skpp.normalize(mat, axis=1)
    sim = mat_row_norm.dot(mat_row_norm.T)
    return sim


def filter_similarity_matrix(sim, sigma):
    """
    filter value by threshold = mean(sim) + sigma * std(sim)
    :param sim: similarity matrix
    :param sigma: hyperparameter for filtering values
    :return: filtered similarity matrix
    """
    sim_mean = np.mean(sim.data)
    sim_std = np.std(sim.data)
    threshold = sim_mean + sigma * sim_std
    sim.data *= sim.data >= threshold  # filter values by threshold
    sim.eliminate_zeros()
    return sim


def get_similarity_matrix(mat, metric=None):
    """
    get similarity matrix of nodes in specified metric
    :param mat: scipy.sparse matrix (csc, csr or coo)
    :param metric: similarity metric
    :return: similarity matrix of nodes
    """
    if metric == 'jaccard':
        return jaccard_similarity(mat.tocsc())
    elif metric == 'cosine':
        return cosine_similarity(mat.tocsc())
    else:
        raise ValueError('Please specify the type of similarity metric.')


def power_method(G, c=0.85, maxiter=100, tol=1e-3, personalization=None):
    """
    r = cWr + (1-c)e
    :param G: Networkx DiGraph created by transition matrix W
    :param c: damping factor
    :param maxiter: maximum number of iterations
    :param tol: error tolerance
    :param personalization: personalization for teleporation vector, uniform distribution if None
    :param print_msg: boolean to check whether to print number of iterations for convergence or not.
    :return: PageRank vector
    """
    nnodes = G.number_of_nodes()
    if personalization is None:
        e = dict.fromkeys(G, 1.0 / nnodes)
    else:
        e = dict.fromkeys(G, 0.0)
        for i in e:
            e[i] = personalization[i, 0]

    r = deepcopy(e)
    for niter in range(maxiter):
        rlast = r
        r = dict.fromkeys(G, 0)
        for n in r:
            for nbr in G[n]:
                r[n] += c * rlast[nbr] * G[n][nbr]['weight']
            r[n] += (1.0 - c) * e[n]
        err = sum([abs(r[n] - rlast[n]) for n in r])
        if err < tol:
            return r

    return r


def revised_power_method(G, c=0.85, alpha=1.0, maxiter=100, tol=1e-3, personalization=None):
    """
    r = Wr + (1-c)/(1+alpha) e
    :param G: Networkx DiGraph created by transition matrix W
    :param c: damping factor
    :param maxiter: maximum number of iterations
    :param tol: error tolerance
    :param personalization: personalization for teleporation vector, uniform distribution if None
    :return: PageRank vector
    """
    nnodes = G.number_of_nodes()
    if personalization is None:
        e = dict.fromkeys(G, 1.0 / nnodes)
    else:
        e = dict.fromkeys(G, 0.0)
        for i in e:
            e[i] = personalization[i, 0]

    r = deepcopy(e)
    for niter in range(maxiter):
        rlast = r
        r = dict.fromkeys(G, 0)
        for n in r:
            for nbr in G[n]:
                r[n] += rlast[nbr] * G[n][nbr]['weight']
            r[n] += (1.0 - c) * e[n] / (1.0 + alpha)
        err = sum([abs(r[n] - rlast[n]) for n in r])
        if err < tol:
            return r
    return r


def reverse_power_method(G, c=0.85, maxiter=100, tol=1e-3, personalization=None):
    """
    r = cr'W + (1-c)e
    :param G: Networkx DiGraph created by transition matrix W
    :param c: damping factor
    :param maxiter: maximum number of iterations
    :param tol: error tolerance
    :param personalization: personalization for teleporation vector, uniform distribution if None
    :param print_msg: boolean to check whether to print number of iterations for convergence or not.
    :return: PageRank vector
    """
    nnodes = G.number_of_nodes()
    if personalization is None:
        e = dict.fromkeys(G, 1.0 / nnodes)
    else:
        e = dict.fromkeys(G, 0.0)
        for i in e:
            e[i] = personalization[i, 0]

    r = deepcopy(e)
    for niter in range(maxiter):
        rlast = r
        r = dict.fromkeys(G, 0)
        for n in r:
            for nbr in G[n]:
                r[nbr] += c * rlast[n] * G[n][nbr]['weight']
            r[n] += (1.0 - c) * e[n]
        err = sum([abs(r[n] - rlast[n]) for n in r])
        if err < tol:
            return r
    return r


# def alias_setup(probs):
#     """
#     Compute utility lists for non-uniform sampling from discrete distributions.
#     Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
#     for details
#     """
#     K = len(probs)
#     q = np.zeros(K)
#     J = np.zeros(K, dtype=np.int)

#     smaller = []
#     larger = []
#     for kk, prob in enumerate(probs):
#         q[kk] = K * prob
#         if q[kk] < 1.0:
#             smaller.append(kk)
#         else:
#             larger.append(kk)

#     while len(smaller) > 0 and len(larger) > 0:
#         small = smaller.pop()
#         large = larger.pop()

#         J[small] = large
#         q[large] = q[large] + q[small] - 1.0
#         if q[large] < 1.0:
#             smaller.append(large)
#         else:
#             larger.append(large)

#     return J, q


# def alias_draw(J, q):
#     """
#     Draw sample from a non-uniform discrete distribution using alias sampling.
#     """
#     K = len(J)

#     kk = int(np.floor(np.random.rand() * K))
#     if np.random.rand() < q[kk]:
#         return kk
#     else:
#         return J[kk]


# Convert sparse matrix to tuple
def sparse_to_tuple(mat):
    if not isspmatrix_coo(mat):
        mat = mat.tocoo()
    coords = np.vstack((mat.row, mat.col)).transpose()
    values = mat.data
    shape = mat.shape
    return coords, values, shape


def train_val_test_split(A, test_frac=.1, val_frac=.05, prevent_disconnect=False, is_directed=False):
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    result = dict()

    # graph should not have diagonal values
    if is_directed:
        G = nx.from_scipy_sparse_matrix(A, create_using=nx.DiGraph(), edge_attribute='weight')
    else:
        G = nx.from_scipy_sparse_matrix(A, create_using=nx.Graph(), edge_attribute='weight')
    num_cc = nx.number_connected_components(G)

    A_triu = triu(A) # upper triangular portion of adj matrix
    A_tuple = sparse_to_tuple(A_triu) # (coords, values, shape), edges only 1 way
    edges = A_tuple[0] # all edges, listed only once (not 2 ways)
    num_test = int(np.floor(edges.shape[0] * test_frac)) # controls how large the test set should be
    num_val = int(np.floor(edges.shape[0] * val_frac)) # controls how alrge the validation set should be

    # Store edges in list of ordered tuples (node1, node2) where node1 < node2
    edge_tuples = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in edges]
    all_edge_tuples = set(edge_tuples)
    train_edges = set(edge_tuples) # initialize train_edges to have all edges
    test_edges, val_edges = set(), set()

    # Iterate over shuffled edges, add to train/val sets
    np.random.shuffle(edge_tuples)
    for edge in edge_tuples:
        node1, node2 = edge[0], edge[1]

        # If removing edge would disconnect a connected component, backtrack and move on
        G.remove_edge(node1, node2)
        if prevent_disconnect == True:
            if nx.number_connected_components(G) > num_cc:
                G.add_edge(node1, node2)
                continue

        # Fill test_edges first
        if len(test_edges) < num_test:
            test_edges.add(edge)
            train_edges.remove(edge)

        # Then, fill val_edges
        elif len(val_edges) < num_val:
            val_edges.add(edge)
            train_edges.remove(edge)

        # Both edge lists full --> break loop
        elif len(test_edges) == num_test and len(val_edges) == num_val:
            break

    if (len(val_edges) < num_val or len(test_edges) < num_test):
        print("WARNING: not enough removable edges to perform full train-test split!")
        print("Num. (test, val) edges requested: ({}, {})".format(num_test, num_val))
        print("Num. (test, val) edges returned: ({}, {})".format(len(test_edges), len(val_edges)))

    if prevent_disconnect == True:
        assert nx.number_connected_components(G) == num_cc

    test_edges_false = set()
    while len(test_edges_false) < num_test:
        idx_i, idx_j = np.random.randint(0, A.shape[0]), np.random.randint(0, A.shape[0])
        if idx_i == idx_j:
            continue
        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))
        # Make sure false_edge not an actual edge, and not a repeat
        if false_edge in all_edge_tuples:
            continue
        if false_edge in test_edges_false:
            continue
        test_edges_false.add(false_edge)

    val_edges_false = set()
    while len(val_edges_false) < num_val:
        idx_i = np.random.randint(0, A.shape[0])
        idx_j = np.random.randint(0, A.shape[0])
        if idx_i == idx_j:
            continue
        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge in not an actual edge, not in test_edges_false, not a repeat
        if false_edge in all_edge_tuples or \
            false_edge in test_edges_false or \
            false_edge in val_edges_false:
            continue
        val_edges_false.add(false_edge)

    train_edges_false = set()
    while len(train_edges_false) < len(train_edges):
        idx_i = np.random.randint(0, A.shape[0])
        idx_j = np.random.randint(0, A.shape[0])
        if idx_i == idx_j:
            continue
        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge in not an actual edge, not in test_edges_false, 
            # not in val_edges_false, not a repeat
        if false_edge in all_edge_tuples or \
            false_edge in test_edges_false or \
            false_edge in val_edges_false or \
            false_edge in train_edges_false:
            continue
        train_edges_false.add(false_edge)

    # assert: false_edges are actually false (not in all_edge_tuples)
    assert test_edges_false.isdisjoint(all_edge_tuples)
    assert val_edges_false.isdisjoint(all_edge_tuples)
    assert train_edges_false.isdisjoint(all_edge_tuples)

    # assert: test, val, train false edges disjoint
    assert test_edges_false.isdisjoint(val_edges_false)
    assert test_edges_false.isdisjoint(train_edges_false)
    assert val_edges_false.isdisjoint(train_edges_false)

    # assert: test, val, train positive edges disjoint
    assert val_edges.isdisjoint(train_edges)
    assert test_edges.isdisjoint(train_edges)
    assert val_edges.isdisjoint(test_edges)

    # Convert edge-lists to numpy arrays
    result['adjacency_train'] = nx.adjacency_matrix(G)
    result['train_edge_pos'] = np.array([list(edge_tuple) for edge_tuple in train_edges])
    result['train_edge_neg'] = np.array([list(edge_tuple) for edge_tuple in train_edges_false])
    result['val_edge_pos'] = np.array([list(edge_tuple) for edge_tuple in val_edges])
    result['val_edge_neg'] = np.array([list(edge_tuple) for edge_tuple in val_edges_false])
    result['test_edge_pos'] = np.array([list(edge_tuple) for edge_tuple in test_edges])
    result['test_edge_neg'] = np.array([list(edge_tuple) for edge_tuple in test_edges_false])

    # NOTE: these edge lists only contain single direction of edge!
    return result
