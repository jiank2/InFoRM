import utils

import numpy as np
import networkx as nx
import sklearn.preprocessing as skpp

from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh


class DebiasModel:
    """
    debiasing the mining model
    """
    def __init__(self):
        return

    def fit(self):
        return

    @staticmethod
    def pagerank(adj, sim, alpha, c=0.85):
        """
        individually fair PageRank
        :param adj: adjacency matrix
        :param sim: similarity matrix
        :param alpha: regularization parameter
        :param c: damping factor
        """
        mat = ((c / (1 + alpha)) * adj) + (((alpha) / (1 + alpha)) * sim)
        graph = nx.from_scipy_sparse_matrix(mat, create_using=nx.Graph())
        r = utils.revised_power_method(graph, c=c, alpha=alpha)
        return r

    @staticmethod
    def spectral_clustering(adj, sim, alpha, ncluster=10, v0=None):
        """
        individually fair spectral clustering
        :param adj: adjacency matrix
        :param sim: similarity matrix
        :param alpha: regularization parameter
        :param ncluster: number of clusters
        :param v0: starting vector for eigen-decomposition
        :return: soft cluster membership matrix of fair spectral clustering
        """
        lap = laplacian(adj) + alpha * laplacian(sim)
        lap *= -1
        _, u = eigsh(lap, which='LM', k=ncluster, sigma=1.0, v0=v0)
        return u

    @staticmethod
    def line(graph, sim, alpha,
             dimension=128, ratio=3200, negative=5,
             init_lr=0.025, batch_size=1000, seed=None):
        """
        individually fair LINE
        :param graph: networkx nx.Graph()
        :param sim: similarity matrix
        :param alpha: regularization hyperparameter
        :param dimension: embedding dimension
        :param ratio: ratio to control edge sampling #sampled_edges = ratio * #nodes
        :param negative: number of negative samples
        :param init_lr: initial learning rate
        :param batch_size: batch size of edges in each training iteration
        :param seed: random seed
        :return: debiased node embeddings
        """
        if seed is not None:
            np.random.seed(seed)

        def _update(vec_u, vec_v, vec_error, label, u, v, lr):
            f = 1 / (1 + np.exp(-np.sum(vec_u * vec_v, axis=1)))
            g = (lr * (label - f)).reshape((len(label), 1))
            vec_error += g * vec_v
            vec_v += g * vec_u
            if label[0] == 1:
                arr = np.asarray(sim[u, v].transpose())
                vec_error -= (2 * lr * alpha * (vec_u - vec_v) * arr)
                vec_v -= (2 * lr * alpha * (vec_v - vec_u) * arr)

        def _train_line():
            nbatch = int(nsamples / batch_size)
            for iter_num in range(nbatch):
                lr = init_lr * max((1 - iter_num * 1.0 / nbatch), 0.0001)
                u, v = [0] * batch_size, [0] * batch_size
                for i in range(batch_size):
                    edge_id = alias_draw(edges_table, edges_prob)
                    u[i], v[i] = edges[edge_id]
                    if not directed and np.random.rand() > 0.5:
                        v[i], u[i] = edges[edge_id]

                vec_error = np.zeros((batch_size, dimension))
                label, target = np.asarray([1 for _ in range(batch_size)]), np.asarray(v)
                for j in range(negative + 1):
                    if j != 0:
                        label = np.asarray([0 for _ in range(batch_size)])
                        for k in range(batch_size):
                            target[k] = alias_draw(nodes_table, nodes_prob)
                    _update(
                        emb_vertex[u], emb_vertex[target], vec_error, label, u, target, lr
                    )
                emb_vertex[u] += vec_error

        directed = nx.is_directed(graph)

        nnodes = graph.number_of_nodes()
        node2id = dict([(node, vid) for vid, node in enumerate(graph.nodes())])

        edges = [[node2id[e[0]], node2id[e[1]]] for e in graph.edges()]
        edge_prob = np.asarray([graph[u][v].get("weight", 1.0) for u, v in graph.edges()])
        edge_prob /= np.sum(edge_prob)
        edges_table, edges_prob = alias_setup(edge_prob)

        degree_weight = np.asarray([0] * nnodes)
        for u, v in graph.edges():
            degree_weight[node2id[u]] += graph[u][v].get("weight", 1.0)
            if not directed:
                degree_weight[node2id[v]] += graph[u][v].get("weight", 1.0)
        node_prob = np.power(degree_weight, 0.75)
        node_prob /= np.sum(node_prob)
        nodes_table, nodes_prob = alias_setup(node_prob)

        nsamples = ratio * nnodes
        emb_vertex = (np.random.random((nnodes, dimension)) - 0.5) / dimension

        # train
        _train_line()

        # normalize
        embeddings = skpp.normalize(emb_vertex, "l2")
        return embeddings


def alias_setup(probs):
    """
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    """
    Draw sample from a non-uniform discrete distribution using alias sampling.
    """
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
