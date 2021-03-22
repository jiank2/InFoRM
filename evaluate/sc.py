import pickle
import json
import load_graph
import utils

import numpy as np
import networkx as nx

from scipy.sparse.csgraph import laplacian
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score


task_lookup = {
    'graph': 'debias the input graph',
    'model': 'debias the mining model',
    'result': 'debias the mining result'
}


def lp_diff(vanilla_result, fair_result, ord=None):
    """
    calculate Lp distance
    :param vanilla_result: vanilla mining result
    :param fair_result: debiased mining result
    :param ord: order (p) of Lp distance
    :return: Lp distance between vanilla mining result and debiased mining result
    """
    diff = 0
    if ord == 'fro':
        for i in range(vanilla_result.shape[1]):
            residual = min(
                np.linalg.norm(vanilla_result[:, i] + fair_result[:, i], ord=2),
                np.linalg.norm(vanilla_result[:, i] - fair_result[:, i], ord=2)
            )
            diff += (residual ** 2)
        return np.sqrt(diff)
    else:
        diff = vanilla_result - fair_result
        return np.linalg.norm(diff, ord=ord)


def nmi(vanilla_result, fair_result, seed=0):
    """
    calculate normalized mutual information (NMI)
    :param vanilla_result: vanilla mining result
    :param fair_result: debiased mining result
    :param seed: random seed
    :return: NMI between vanilla mining result and debiased mining result
    """
    # kmeans for vanilla spectral clustering
    vanilla_kmeans = KMeans(n_clusters=vanilla_result.shape[1], random_state=seed, n_init=1).fit(vanilla_result)
    vanilla_labels = vanilla_kmeans.labels_

    # kmeans for fair spectral clustering
    fair_kmeans = KMeans(n_clusters=fair_result.shape[1], random_state=seed, n_init=1).fit(fair_result)
    fair_labels = fair_kmeans.labels_

    # calculate normalized mutual information
    nmi = normalized_mutual_info_score(vanilla_labels, fair_labels, average_method='arithmetic')
    return nmi


def calc_bias(name, metric, vanilla_result, fair_result):
    """
    calculate bias reduction
    :param name: dataset name
    :param metric: similarity metric (jaccard or cosine)
    :param vanilla_result: vanilla mining result
    :param fair_result: debiased mining result
    :return: bias reduction
    """
    # load graph
    if name == 'ppi':
        data = load_graph.read_mat(name)
        graph = data['graph']
    else:
        graph = load_graph.read_graph(name)
    lcc = max(nx.connected_components(graph), key=len)  # take largest connected components
    adj = nx.to_scipy_sparse_matrix(graph, nodelist=lcc, dtype='float', format='csc')

    # build similarity matrix
    sim = utils.get_similarity_matrix(adj, metric=metric)
    lap = laplacian(sim)

    # calculate bias
    vanilla_bias = utils.trace(vanilla_result.T @ lap @ vanilla_result)  # vanilla bias
    fair_bias = utils.trace(fair_result.T @ lap @ fair_result)  # fair bias
    reduction = 1 - (fair_bias / vanilla_bias)
    return reduction


def evaluate(name, metric, task):
    """
    main function for evaluation
    :param name: dataset name
    :param metric: similarity metric (jaccard or cosine)
    :param task: debiasing task (graph, model or result)
    """
    # scores
    result = dict()

    # load vanilla result
    with open('result/sc/vanilla.pickle', 'rb') as f:
        vanilla = pickle.load(f)

    # load fair result
    with open('result/sc/{}/{}.pickle'.format(task, metric), 'rb') as f:
        fair = pickle.load(f)

    # get vanilla and fair results
    vanilla_result = vanilla[name]['eigenvectors']
    fair_result = fair[name]
            
    # evaluate
    result['dataset'] = name
    result['metric'] = '{} similarity'.format(metric)
    result['task'] = task_lookup[task]
    result['diff'] = lp_diff(vanilla_result, fair_result, ord='fro') / np.linalg.norm(vanilla_result, ord='fro')
    max_val = -1
    for seed in range(21):
        val = nmi(vanilla_result, fair_result, seed=seed)
        if val > max_val:
            max_val = val
        result['nmi'] = max_val
        result['bias'] = calc_bias(name, metric, vanilla_result, fair_result)

    print(result)

    # save to file
    with open('result/sc/{}/evaluation_{}.json'.format(task, metric), 'a') as f:
        json.dump(result, f, indent=4)
        f.write('\n')


# if __name__ == '__main__':
#     evaluate(name='ppi', metric='jaccard', task='graph')
#     evaluate(name='ppi', metric='cosine', task='graph')

#     evaluate(name='ppi', metric='jaccard', task='model')
#     evaluate(name='ppi', metric='cosine', task='model')

#     evaluate(name='ppi', metric='jaccard', task='result')
#     evaluate(name='ppi', metric='cosine', task='result')
