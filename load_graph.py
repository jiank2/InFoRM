import os
import pickle

import networkx as nx
import scipy.io as sio


def read_graph(name, is_directed=False):
    """
    read graph data from edge list file
    :param name: name of the graph
    :param is_directed: if the graph is directed or not
    :return: adjacency matrix, Networkx Graph (if undirected) or DiGraph (if directed)
    """
    if is_directed:
        PATH = os.path.join('data', 'directed', '{}.txt'.format(name))
        G = nx.read_edgelist(PATH, create_using=nx.DiGraph(), nodetype=int, data=(('weight', float),))
    else:
        PATH = os.path.join('data', '{}.txt'.format(name))
        G = nx.read_edgelist(PATH, create_using=nx.Graph(), nodetype=int, data=(('weight', float),))
    return G


def read_mat(name):
    """
    read .mat file
    :param name: dataset name
    :return: a dict containing adjacency matrix, nx.Graph() and its node labels
    """
    result = dict()
    PATH = os.path.join('data', '{}.mat'.format(name))
    matfile = sio.loadmat(PATH)
    result['adjacency'] = matfile['network']
    result['label'] = matfile['group']
    result['graph'] = nx.from_scipy_sparse_matrix(result['adjacency'], create_using=nx.Graph(), edge_attribute='weight')
    return result


def read_pickle(name):
    """
    read .pickle file
    :param name: dataset name
    :return: a dict containing adjacency matrix, nx.Graph() and its node labels
    """
    with open('data/{}.pickle'.format(name), 'rb') as f:
        result = pickle.load(f)
    return result
