import utils

import numpy as np
import networkx as nx

import torch
import torch.nn as nn

from scipy.sparse import csc_matrix, identity, diags
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh, svds


class DebiasGraph:
    """
    debiasing the input graph
    """
    def __init__(self):
        return

    def fit(self):
        return

    @staticmethod
    def pagerank(init_adj, sim, alpha, c=0.85, maxiter=100, lr=0.1, tol=1e-6):
        """
        learning fair graph for PageRank
        :param init_adj: initial adjacency matrix
        :param sim: similarity matrix
        :param alpha: regularization parameter
        :param c: damping factor
        :param maxiter: maximum number of iterations
        :param lr: learning rate
        :param tol: tolerance
        :return: debiased nx.Graph()
        """
        # helper function to calculate partial derivative
        def partial_grad(x, r, src, tgt):
            return 2 * alpha * c * x[src, 0] * r[tgt, 0]

        graph = nx.from_scipy_sparse_matrix(init_adj, create_using=nx.Graph())
        lap = laplacian(sim)
        for niter in range(maxiter):
            # calc low-rank structure
            r = utils.power_method(graph, c=c, maxiter=maxiter)
            r = np.array([list(r.values())])
            r = csc_matrix(np.array(r).transpose())
            tele = lap @ r  # Ls dot r
            vec = utils.reverse_power_method(graph, c=c, personalization=tele, maxiter=maxiter)
            vec = np.array([list(vec.values())])
            vec = csc_matrix(np.array(vec).transpose())

            # iterate each edge to update gradient
            residual = 0
            for (src, tgt) in graph.edges:
                # src, tgt = e
                if src != tgt:
                    norm = 4 * (graph[src][tgt]['weight'] - init_adj[src, tgt])
                    partial = partial_grad(vec, r, src, tgt) + \
                              partial_grad(vec, r, tgt, src)
                    # grad = 4 * (graph[src][tgt]['weight'] - init_adj[src, tgt]) + \
                    #        partial_grad(vec, r, src, tgt) + \
                    #        partial_grad(vec, r, tgt, src)
                else:
                    norm = 2 * (graph[src][tgt]['weight'] - init_adj[src, tgt])
                    partial = partial_grad(vec, r, src, tgt)
                    # grad = 2 * (graph[src][tgt]['weight'] - init_adj[src, tgt]) + \
                    #        partial_grad(vec, r, src, tgt)
                grad = norm + partial
                if graph[src][tgt]['weight'] >= lr * grad:
                    graph[src][tgt]['weight'] -= lr * grad
                    residual += (grad ** 2)

            if np.sqrt(residual) < tol:
                return graph
        return graph

    @staticmethod
    def spectral_clustering(init_adj, sim, alpha, ncluster=10, v0=None, maxiter=100, lr=0.1, tol=1e-6):
        """
        learning fair graph for spectral clustering
        :param init_adj: initial adjacency matrix
        :param sim: similarity matrix
        :param alpha: regularization parameter
        :param ncluster: number of clusters
        :param v0: starting vector for eigen-decomposition
        :param maxiter: maximum number of iterations
        :param lr: learning rate
        :param tol: tolerance
        :return: debiased adjacency matrix
        """
        # helper function to calculate partial derivative
        def partial_grad(x, u, src, tgt):
            return 2 * alpha * (x[src, :] @ u[src, :] - x[src, :] @ u[tgt, :])

        adj = init_adj.copy().tocoo()
        adj_row, adj_col = adj.row, adj.col
        adj = adj.tocsc()

        lap = laplacian(sim)

        nedges = adj.nnz

        for niter in range(maxiter):
            try:
                lap_adj = laplacian(adj)
                lap_adj *= -1
                v, u = eigsh(lap_adj, which='LM', k=ncluster, sigma=1.0, v0=v0)
            except:
                return None
            x = u.copy()

            # calc low-rank structure
            try:
                for i in range(ncluster):
                    ui = u[:, [i]]
                    m = v[i] * identity(adj.shape[0]) - lap_adj
                    m_u, m_sigma, m_vt = svds(m)  # m_vt is the transpose of m_v
                    m_sigma_pinv = diags(np.divide(1.0, m_sigma))
                    xcol = lap @ ui
                    xcol = m_u.T @ xcol
                    xcol = m_sigma_pinv @ xcol
                    xcol = m_vt.T @ xcol  # this is equivalent to VM @ x
                    x[:, [i]] = xcol
            except:
                break

            # iterate each edge to update gradient
            residual, idx = 0, 0
            for idx in range(nedges):
                src, tgt = adj_row[idx], adj_col[idx]
                if src > tgt:
                    continue
                if src != tgt:
                    norm = 4 * (adj[src, tgt] - init_adj[src, tgt])
                    partial = partial_grad(x, u, src, tgt) + \
                              partial_grad(x, u, tgt, src)
                    # grad = 4 * (adj[src, tgt] - init_adj[src, tgt]) + \
                    #        partial_grad(x, u, src, tgt, alpha) + \
                    #        partial_grad(x, u, tgt, src, alpha)
                else:
                    norm = 2 * (adj[src, tgt] - init_adj[src, tgt])
                    partial = partial_grad(x, u, src, tgt)
                    # grad = 2 * (adj[src, tgt] - init_adj[src, tgt]) + \
                    #        partial_grad(x, u, src, tgt, alpha)
                grad = norm + partial
                if adj[src, tgt] >= lr * grad:
                    adj[src, tgt] -= lr * grad
                    residual += (grad ** 2)
                    if src != tgt:
                        adj[tgt, src] -= lr * grad
                        residual += (grad ** 2)

            if np.sqrt(residual) < tol:
                return adj
        return adj

    @staticmethod
    def line(init_adj, sim, alpha, maxiter=100, lr=0.1, tol=1e-6):
        """
        learning fair graph for LINE
        :param init_adj: initial adjacency matrix
        :param sim: similarity matrix
        :param alpha: regularization parameter
        :param maxiter: maximum number of iterations
        :param lr: learning rate
        :param tol: tolerance
        :return: debiased adjacency matrix
        """
        # helper function to calculate partial derivative
        def partial_grad(adj, lap, vec, src, tgt):
            first = lap[src, tgt] / (adj[src, tgt] + adj[tgt, src])  # f(A+A') * LS
            second = vec[src]  # diag(BLs)
            partial_grad = alpha * (first - second)
            return 2 * partial_grad

        adj = init_adj.copy().tocoo()
        adj_row, adj_col = adj.row, adj.col
        adj = adj.tocsc()

        lap = laplacian(sim).tocoo()
        lap_row, lap_col = lap.row, lap.col
        lap = lap.tocsc()

        nsims, nedges = lap.nnz, adj.nnz
        for niter in range(maxiter):
            # calculate first column of diag(BLs)1, which is diag(BLs)
            vec = np.asarray([0.0] * init_adj.shape[0])  # vec = diag(BLs)
            d = adj.sum(axis=1).T
            d025, d075, d125 = np.power(d, 0.25), np.power(d, 0.75), np.power(d, 1.25)
            for idx in range(nsims):
                src, tgt = lap_row[idx], lap_col[idx]
                if (src > tgt) or (d[0, tgt] == 0) or (d[0, src] == 0):
                    continue
                denom1 = 4.0 * ((d125[0, tgt] / d025[0, src]) + d[0, tgt])
                first = 3.0 / denom1
                denom2 = (d075[0, tgt] * d025[0, src]) + d[0, tgt]
                second = 1.0 / denom2
                val_tgt_src = first + second
                vec[tgt] += float(val_tgt_src * lap[src, tgt])
                if src != tgt:
                    denom1 = 4.0 * ((d125[0, src] / d025[0, tgt]) + d[0, src])
                    first = 3.0 / denom1
                    denom2 = (d075[0, src] * d025[0, tgt]) + d[0, src]
                    second = 1.0 / denom2
                    val_src_tgt = first + second
                    vec[src] += float(val_src_tgt * lap[tgt, src])

            # iterate each edge to update gradient
            residual = 0
            for idx in range(nedges):
                src, tgt = adj_row[idx], adj_col[idx]
                if src > tgt:
                    continue
                if src != tgt:
                    norm = 4 * (adj[src, tgt] - init_adj[src, tgt])
                    partial = partial_grad(adj, lap, vec, src, tgt) + \
                              partial_grad(adj, lap, vec, tgt, src)
                else:
                    norm = 2 * (adj[src, tgt] - init_adj[src, tgt])
                    partial = partial_grad(adj, lap, vec, src, tgt)
                grad = norm + partial

                if adj[src, tgt] >= lr * grad:
                    adj[src, tgt] -= lr * grad
                    residual += (grad ** 2)
                    if src != tgt:
                        adj[tgt, src] -= lr * grad
                        residual += (grad ** 2)

            if np.sqrt(residual) < tol:
                return adj
        return adj
