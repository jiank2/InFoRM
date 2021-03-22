import pickle
import json
import load_graph
import utils

import numpy as np
import networkx as nx

from scipy.sparse.csgraph import laplacian


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
    diff = vanilla_result - fair_result
    return np.linalg.norm(diff, ord=ord)


def kl_divergence(vanilla_result, fair_result):
    """
    calculate KL divergence
    :param vanilla_result: vanilla mining result
    :param fair_result: debiased mining result
    :return: KL divergence between vanilla mining result and debiased mining result
    """
    norm_vanilla_result = np.sum(vanilla_result)
    norm_fair_result = np.sum(fair_result)
    kl = 0
    for i in range(vanilla_result.shape[0]):
        x = fair_result[i] / norm_fair_result
        y = vanilla_result[i] / norm_vanilla_result
        if y != 0 and x != 0:
            kl += x * np.log(x / y)
    return kl


def precision_at_k(vanilla_topk, fair_topk):
    """
    calculate precision @ K
    :param vanilla_topk: top K nodes in vanilla mining result
    :param fair_topk: top K nodes in debiased mining result
    :return: precision @ K
    """
    topk = set(fair_topk)
    groundtruth = set(vanilla_topk)
    return len(topk.intersection(groundtruth)) / len(topk)


def calc_single_dcg(rel, pos):
    """
    calculate DCG for a single node
    :param rel: relevance (0 or 1) of this node
    :param pos: ranking of this node
    :return: DCG of this node
    """
    numerator = (2 ** rel) - 1
    denominator = np.log(1 + pos)
    return numerator / denominator


def ndcg_at_k(vanilla_topk, fair_topk):
    """
    calculate NDCG @ K
    :param vanilla_topk: top K nodes in vanilla mining result
    :param fair_topk: top K nodes in debiased mining result
    :return: NDCG @ K
    """
    dcg, idcg = 0, 0
    for i in range(len(fair_topk)):
        if fair_topk[i] in vanilla_topk:
            dcg += calc_single_dcg(1, i + 1)
        idcg += calc_single_dcg(1, i + 1)
    return dcg / idcg


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
        adj = data['adjacency']
    else:
        graph = load_graph.read_graph(name)
        adj = nx.to_scipy_sparse_matrix(graph, dtype='float', format='csc')
    adj = utils.symmetric_normalize(adj)

    # build similarity matrix
    sim = utils.filter_similarity_matrix(utils.get_similarity_matrix(adj, metric=metric), sigma=0.75)
    sim = utils.symmetric_normalize(sim)
    lap = laplacian(sim)

    # calculate bias
    vanilla_bias = utils.trace(vanilla_result.T @ lap @ vanilla_result)
    fair_bias = utils.trace(fair_result.T @ lap @ fair_result)
    reduction = 1 - (fair_bias/vanilla_bias)
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
    with open('result/pagerank/vanilla.pickle', 'rb') as f:
        vanilla = pickle.load(f)

    # load fair result
    with open('result/pagerank/{}/{}.pickle'.format(task, metric), 'rb') as f:
        fair = pickle.load(f)

    # get vanilla and fair results
    vanilla_result = np.asarray(vanilla[name].todense()).flatten()  # vanilla result, flatten to np.array
    fair_result = np.asarray(fair[name].todense()).flatten()  # fair result, flatten to np.array

    # evaluate
    result['dataset'] = name
    result['metric'] = '{} similarity'.format(metric)
    result['task'] = task_lookup[task]
    result['diff'] = lp_diff(vanilla_result, fair_result, ord=2) / np.linalg.norm(vanilla_result, ord=2)
    result['kl'] = kl_divergence(vanilla_result, fair_result)
    result['precision'] = dict()
    result['ndcg'] = dict()

    k = 50
    vanilla_topk = np.argsort(vanilla_result)[-k:][::-1]
    fair_topk = np.argsort(fair_result)[-k:][::-1]
    result['precision'][k] = precision_at_k(vanilla_topk, fair_topk)
    result['ndcg'][k] = ndcg_at_k(vanilla_topk, fair_topk)

    result['bias'] = calc_bias(name, metric, vanilla[name], fair[name])

    print(result)

    # save to file
    with open('result/pagerank/{}/evaluation_{}.json'.format(task, metric), 'a') as f:
        json.dump(result, f, indent=4)
        f.write('\n')


# if __name__ == '__main__':
#     evaluate(name='ppi', metric='jaccard', task='graph')
#     evaluate(name='ppi', metric='cosine', task='graph')

#     evaluate(name='ppi', metric='jaccard', task='model')
#     evaluate(name='ppi', metric='cosine', task='model')

#     evaluate(name='ppi', metric='jaccard', task='result')
#     evaluate(name='ppi', metric='cosine', task='result')