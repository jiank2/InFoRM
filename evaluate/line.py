import pickle
import json
import load_graph
import utils

import numpy as np
import networkx as nx 

from scipy.sparse.csgraph import laplacian
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score


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


def get_score(embs, src, tgt):
    """
    calculate score for link prediction
    :param embs: embedding matrix
    :param src: source node
    :param tgt: target node
    """
    vec_src = embs[int(src)]
    vec_tgt = embs[int(tgt)]
    return np.dot(vec_src, vec_tgt) / (np.linalg.norm(vec_src) * np.linalg.norm(vec_tgt))


def link_prediction(data, embs):
    """
    link prediction
    :param data: input data
    :param embs: embedding matrix
    """
    true_edges = data['test_edge_pos']
    false_edges = data['test_edge_neg']

    true_list = list()
    prediction_list = list()
    for src, tgt in true_edges:
        true_list.append(1)
        prediction_list.append(get_score(embs, src, tgt))

    for src, tgt in false_edges:
        true_list.append(0)
        prediction_list.append(get_score(embs, src, tgt))

    sorted_pred = prediction_list[:]
    sorted_pred.sort()
    threshold = sorted_pred[-len(true_edges)]

    ypred = np.zeros(len(prediction_list), dtype=np.int32)
    for i in range(len(prediction_list)):
        if prediction_list[i] >= threshold:
            ypred[i] = 1

    ytrue = np.array(true_list)
    yscores = np.array(prediction_list)
    precision, recall, _ = precision_recall_curve(ytrue, yscores)
    return roc_auc_score(ytrue, yscores), f1_score(ytrue, ypred)


def calc_bias(data, metric, vanilla_result, fair_result):
    """
    calculate bias reduction
    :param data: input data
    :param metric: similarity metric (jaccard or cosine)
    :param vanilla_result: vanilla mining result
    :param fair_result: debiased mining result
    :return: a dict containing bias reductions of training edges, validation edges and test edges
    """
    biases = dict()
    # load graph
    adj_train = data['adjacency_train']
    # build similarity matrix
    sim = utils.filter_similarity_matrix(
        utils.get_similarity_matrix(adj_train, metric=metric), sigma=0.75
    )
    lap = laplacian(sim)

    # calculate training bias
    biases['train'] = [
        utils.trace(vanilla_result.T @ lap @ vanilla_result),
        utils.trace(fair_result.T @ lap @ fair_result)
    ]
    reduction = 1 - (biases['train'][1] / biases['train'][0])
    biases['train'].clear()
    biases['train'].append(reduction)

    # calculating bias on validation set
    val_bias = [0, 0]  # first one is vanilla, second one is fair
    for src, tgt in data['val_edge_pos']:
        vanilla_edge_bias = (lp_diff(vanilla_result[src], vanilla_result[tgt], ord=2) ** 2) * sim[src, tgt]
        fair_edge_bias = (lp_diff(fair_result[src], fair_result[tgt], ord=2) ** 2) * sim[src, tgt]
        val_bias[0] += vanilla_edge_bias
        val_bias[1] += fair_edge_bias
    for src, tgt in data['val_edge_neg']:
        vanilla_edge_bias = (lp_diff(vanilla_result[src], vanilla_result[tgt], ord=2) ** 2) * sim[src, tgt]
        fair_edge_bias = (lp_diff(fair_result[src], fair_result[tgt], ord=2) ** 2) * sim[src, tgt]
        val_bias[0] += vanilla_edge_bias
        val_bias[1] += fair_edge_bias
    reduction = 1 - (val_bias[1] / val_bias[0])
    val_bias.clear()
    val_bias.append(reduction)
    biases['validation'] = val_bias

    # calculating bias on test set
    test_bias = [0, 0]  # first one is vanilla, second one is fair
    for src, tgt in data['test_edge_pos']:
        vanilla_edge_bias = (lp_diff(vanilla_result[src], vanilla_result[tgt], ord=2) ** 2) * sim[src, tgt]
        fair_edge_bias = (lp_diff(fair_result[src], fair_result[tgt], ord=2) ** 2) * sim[src, tgt]
        test_bias[0] += vanilla_edge_bias
        test_bias[1] += fair_edge_bias
    for src, tgt in data['test_edge_neg']:
        vanilla_edge_bias = (lp_diff(vanilla_result[src], vanilla_result[tgt], ord=2) ** 2) * sim[src, tgt]
        fair_edge_bias = (lp_diff(fair_result[src], fair_result[tgt], ord=2) ** 2) * sim[src, tgt]
        test_bias[0] += vanilla_edge_bias
        test_bias[1] += fair_edge_bias
    reduction = 1 - (test_bias[1] / test_bias[0])
    test_bias.clear()
    test_bias.append(reduction)
    biases['test'] = test_bias

    return biases


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
    with open('result/line/vanilla.pickle', 'rb') as f:
        vanilla = pickle.load(f)

    # load fair result
    with open('result/line/{}/{}.pickle'.format(task, metric), 'rb') as f:
        fair = pickle.load(f)

    # load link prediction data
    data = load_graph.read_pickle(name)

    # get vanilla and fair results
    vanilla_result = vanilla[name]
    fair_result = fair[name]
            
    # evaluate
    result['dataset'] = name
    result['metric'] = '{} similarity'.format(metric)
    result['task'] = task_lookup[task]
    result['diff'] = lp_diff(vanilla_result, fair_result, ord='fro') / np.linalg.norm(vanilla_result, ord='fro')
    vanilla_scores = link_prediction(data, vanilla_result)
    fair_scores = link_prediction(data, fair_result)
    result['roc-auc'] = vanilla_scores[0], fair_scores[0]
    result['f1'] = vanilla_scores[1], fair_scores[1]
    result['bias'] = calc_bias(data, metric, vanilla_result, fair_result)
    
    print(result)

    # save to file
    with open('result/line/{}/evaluation_{}.json'.format(task, metric), 'a') as f:
        json.dump(result, f, indent=4)
        f.write('\n')


# if __name__ == '__main__':
#     evaluate(name='ppi', metric='jaccard', task='graph')
#     evaluate(name='ppi', metric='cosine', task='graph')

#     evaluate(name='ppi', metric='jaccard', task='model')
#     evaluate(name='ppi', metric='cosine', task='model')

#     evaluate(name='ppi', metric='jaccard', task='result')
#     evaluate(name='ppi', metric='cosine', task='result')
