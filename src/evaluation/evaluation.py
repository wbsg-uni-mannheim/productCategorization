import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, f1_score
from networkx import all_pairs_shortest_path_length, relabel_nodes
from contextlib import contextmanager
from sklearn.preprocessing import MultiLabelBinarizer
import logging


def testing_traditional(y_true, y_pred, all_labels):
    if all_labels:
        return precision_recall_fscore_support(y_true, y_pred, pos_label=None, average='weighted')
    else:
        return precision_recall_fscore_support(y_true, y_pred, pos_label=None, labels=np.unique(y_true),
                                               average='weighted')


def eval_traditional(name, y_true, y_pred):
    logger = logging.getLogger(__name__)
    if name is None:
        scores_all_labels_false = testing_traditional(y_true, y_pred, all_labels=False)
        return scores_all_labels_false[:3]
    else:
        scores_all_labels_false = testing_traditional(y_true, y_pred, all_labels=False)
        scores_all_labels_true = testing_traditional(y_true, y_pred, all_labels=True)
        f1_score_all_labels_false = f1_score(y_true, y_pred, average='macro', labels=np.unique(y_true))
        f1_score_all_labels_true = f1_score(y_true, y_pred, average='macro')

        logger.info(
            "{} | prec_weighted: {:4f} | rec_weighted: {:4f} | f1_weighted: {:4f} | f1_macro: {:4f} | true labels".format(
                name, scores_all_labels_false[0], scores_all_labels_false[1], scores_all_labels_false[2],
                f1_score_all_labels_false))

        logger.info(
            "{} | prec_weighted: {:4f} | rec_weighted: {:4f} | f1_weighted: {:4f} | f1_macro: {:4f} | all labels".format(
                name, scores_all_labels_true[0], scores_all_labels_true[1], scores_all_labels_true[2],
                f1_score_all_labels_true))

        return [scores_all_labels_false, f1_score_all_labels_false], [scores_all_labels_true, f1_score_all_labels_true]


def f_1_weighted(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=np.unique(y_true), average='weighted')


def h_precision_score(y_true, y_pred, class_hierarchy, root):
    """ Influenced by https://github.com/asitang/sklearn-hierarchical-classification/blob/develop/sklearn_hierarchical/metrics.py """
    y_true_ = fill_ancestors(y_true, root, graph=class_hierarchy)
    y_pred_ = fill_ancestors(y_pred, root, graph=class_hierarchy)

    # print('y_true_',y_true_[64])
    # print('y_pred_',y_pred_[64])

    ix = np.where((y_true_ != 0) & (y_pred_ != 0))

    true_positives = len(ix[0])
    all_results = np.count_nonzero(y_pred_)

    # print('all_results', all_results)
    if all_results > 0:
        return true_positives / all_results
    else:
        return 0


def h_recall_score(y_true, y_pred, class_hierarchy, root):
    """ Influenced by https://github.com/asitang/sklearn-hierarchical-classification/blob/develop/sklearn_hierarchical/metrics.py """
    y_true_ = fill_ancestors(y_true, root, graph=class_hierarchy)
    y_pred_ = fill_ancestors(y_pred, root, graph=class_hierarchy)

    ix = np.where((y_true_ != 0) & (y_pred_ != 0))

    true_positives = len(ix[0])
    all_positives = np.count_nonzero(y_true_)
    # print('all_positives',all_positives)
    if all_positives > 0:
        return true_positives / all_positives
    else:
        return 0


def h_fbeta_score(y_true, y_pred, class_hierarchy, root, beta=1.):
    """ Influenced by https://github.com/asitang/sklearn-hierarchical-classification/blob/develop/sklearn_hierarchical/metrics.py """
    hP = h_precision_score(y_true, y_pred, class_hierarchy, root)
    hR = h_recall_score(y_true, y_pred, class_hierarchy, root)
    if (beta ** 2. * hP + hR) > 0:
        return (1. + beta ** 2.) * hP * hR / (beta ** 2. * hP + hR)
    else:
        return 0


@contextmanager
def multi_labeled(y_true, y_pred, graph, root):
    """ Influenced by https://github.com/asitang/sklearn-hierarchical-classification/blob/develop/sklearn_hierarchical/metrics.py """
    mlb = MultiLabelBinarizer()
    all_classes = [
        node
        for node in graph.nodes
        if node != root
    ]
    # print('all_classes',all_classes)
    # Nb. we pass a (singleton) list-within-a-list as fit() expects an iterable of iterables -> Changed implementation here
    all_classes_new = []
    for klasse in all_classes:
        all_classes_new.append([klasse])

    mlb.fit(all_classes_new)

    y_true_new = []
    for klasse in y_true:
        y_true_new.append([klasse])

    y_pred_new = []
    for klasse in y_pred:
        y_pred_new.append([klasse])

    node_label_mapping = {
        old_label: new_label
        for new_label, old_label in enumerate(list(mlb.classes_))
    }
    # print('node_label_mapping',node_label_mapping)
    # print('y_true transform',y_true)
    # print('mlb.transform(y_true)',mlb.transform(y_true_new)[0])
    yield (
        mlb.transform(y_true_new),
        mlb.transform(y_pred_new),
        relabel_nodes(graph, node_label_mapping),
        root,
    )


def hierarchical_eval(name, y_true, y_pred, tree, root):
    logger = logging.getLogger(__name__)
    with multi_labeled(y_true, y_pred, tree, root) as (y_test_, y_pred_, graph_, root_):
        h_fbeta = h_fbeta_score(
            y_test_,
            y_pred_,
            graph_,
            root_,
        )
        if not name:
            return h_fbeta
        else:
            logger.info("{} | h_f1: {:4f}".format(name, h_fbeta))
            return h_fbeta


def fill_ancestors(y, root, graph, copy=True):
    """ Influenced by https://github.com/asitang/sklearn-hierarchical-classification/blob/develop/sklearn_hierarchical/metrics.py """
    y_ = y.copy() if copy else y
    paths = all_pairs_shortest_path_length(graph.reverse(copy=False))
    for target, distances in paths:
        if target == root:
            # Our stub ROOT node, can skip
            continue
        # print('y',y)
        # print('target',target)
        ix_rows = np.where(y[:, target] > 0)[0]
        # all ancestors, except the last one which would be the root node
        ancestors = list(distances.keys())[:-1]
        # print('ancestors',ancestors)
        # print('mesh',tuple(np.meshgrid(ix_rows, ancestors)))
        # print('target',target)
        # print(type(target))
        y_[tuple(np.meshgrid(ix_rows, ancestors))] = 1
    graph.reverse(copy=False)
    # print('y_',y_[0])
    return y_


# only if SelectKBest was used
def get_most_important_features(classifier_pipeline_object):
    feature_names = classifier_pipeline_object['vect'].get_feature_names()
    # print(feature_names)
    return [feature_names[i] for i in classifier_pipeline_object['chi'].get_support(indices=True)]

# Not sure if this part of code is executed
# def print_top10(classifier_pipeline_object):
#    feature_names = classifier_pipeline_object['vect'].get_feature_names()
#    for i, class_label in enumerate(classifier_pipeline_object['clf'].classes_):
#        top10 = np.argsort(classifier_pipeline_object['clf'].coef_[i])[-10:]
#        print("%s: %s" % (class_label,
#              " ".join(feature_names[j] for j in top10)))
#        print()

class TransformersEvaluator():
    def __init__(self, dataset_name, experiment_name):
        project_dir = Path(__file__).resolve().parents[2]
        path_to_tree = project_dir.joinpath('data', 'raw', dataset_name, 'tree', 'tree_{}.pkl'.format(dataset_name))

        with open(path_to_tree, 'rb') as f:
            self.tree = pickle.load(f)

        self.root = [node[0] for node in self.tree.in_degree if node[1] == 0][0]

    def compute_metrics_transformers(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        return self.compute_metrics(labels,preds)


    def compute_metrics(self, labels, preds):
        scores_all_labels = eval_traditional("bert", labels, preds)
        h_f_score = hierarchical_eval("bert", labels, preds, self.tree, self.root)

        # To-Do: Change structure of elements!
        return {'weighted_prec': scores_all_labels[0][0][0],
                'weighted_rec': scores_all_labels[0][0][1],
                'weighted_f1': scores_all_labels[0][0][2],
                'macro_f1': scores_all_labels[0][1],
                'h_f1': h_f_score}