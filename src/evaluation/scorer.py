from sklearn.metrics import precision_recall_fscore_support, f1_score
from networkx import all_pairs_shortest_path_length, relabel_nodes
from contextlib import contextmanager
from sklearn.preprocessing import MultiLabelBinarizer
import logging
import numpy as np


def score_traditional(gs: list, prediction: list, name='Unknown'):
    logger = logging.getLogger(__name__)
    logger.debug(gs)
    logger.debug(prediction)


    w_prec, w_rec, w_f1, support = precision_recall_fscore_support(gs, prediction, average='weighted', zero_division=0)
    f1_macro = f1_score(gs, prediction, average='macro')

    logger.info(
        "{} - Leaf Nodes: | prec_weighted: {:4f} | rec_weighted: {:4f} | f1_weighted: {:4f} | f1_macro: {:4f}".format(
            name, w_prec, w_rec, w_f1, f1_macro))

    return [w_prec, w_rec, w_f1, f1_macro]  # Precision_weighted, Recall_weighted, F1_weighted, F1_macro


def score_mwpd(gs: list, prediction: list):
    logger = logging.getLogger(__name__)
    logger.debug(gs)
    logger.debug(prediction)
    macro_scores = precision_recall_fscore_support(gs, prediction, average='macro', zero_division=0)
    weighted_scores = precision_recall_fscore_support(gs, prediction, average='weighted', zero_division=0)
    return {'weighted': weighted_scores, 'macro': macro_scores}


# Unsure if this is needed!
# def f_1_weighted(y_true, y_pred):
#    return f1_score(y_true, y_pred, labels=np.unique(y_true), average='weighted')


def h_score(y_true, y_pred, class_hierarchy, root):
    """ Influenced by https://github.com/asitang/sklearn-hierarchical-classification/blob/develop
    /sklearn_hierarchical/metrics.py """
    y_true_ = fill_ancestors(y_true, root, graph=class_hierarchy)
    y_pred_ = fill_ancestors(y_pred, root, graph=class_hierarchy)

    ix = np.where((y_true_ != 0) & (y_pred_ != 0))

    true_positives = len(ix[0])
    all_positives = np.count_nonzero(y_true_)
    all_results = np.count_nonzero(y_pred_)

    h_precision = 0
    if all_results > 0:
        h_precision = true_positives / all_results

    h_recall = 0
    if all_positives > 0:
        h_recall = true_positives / all_positives

    return h_precision, h_recall


def h_fbeta_score(y_true, y_pred, class_hierarchy, root, beta=1.):
    """ Influenced by https://github.com/asitang/sklearn-hierarchical-classification/blob/develop
    /sklearn_hierarchical/metrics.py """
    hP, hR = h_score(y_true, y_pred, class_hierarchy, root)
    if (beta ** 2. * hP + hR) > 0:
        return (1. + beta ** 2.) * hP * hR / (beta ** 2. * hP + hR)
    else:
        return 0


@contextmanager
def multi_labeled(y_true, y_pred, graph, root):
    """ Influenced by https://github.com/asitang/sklearn-hierarchical-classification/blob/develop
    /sklearn_hierarchical/metrics.py """
    mlb = MultiLabelBinarizer()
    all_classes = [
        node
        for node in graph.nodes
        if node != root
    ]
    # print('all_classes',all_classes) Nb. we pass a (singleton) list-within-a-list as fit() expects an iterable of
    # iterables -> Changed implementation here
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


def hierarchical_score(y_true, y_pred, tree, root, name='Unknown'):
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
            logger.info("{} - Hierarchy: | h_f1: {:4f}".format(name, h_fbeta))
            return h_fbeta


def fill_ancestors(y, root, graph, copy=True):
    """ Influenced by https://github.com/asitang/sklearn-hierarchical-classification/blob/develop
    /sklearn_hierarchical/metrics.py """
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


class HierarchicalScorer:
    def __init__(self, experiment_name, tree, transformer_decoder=None, num_labels_per_level=None):
        self.logger = logging.getLogger(__name__)

        self.experiment_name = experiment_name
        self.tree = tree
        self.transformer_decoder = transformer_decoder
        self.num_labels_per_lvl = num_labels_per_level

        self.root = [node[0] for node in self.tree.in_degree if node[1] == 0][0]

    def determine_path_to_root(self, nodes):
        predecessors = [k for k in self.tree.predecessors(nodes[-1])]
        if len(predecessors) > 0:
            predecessor = predecessors[0]
        else:
            predecessor = self.root #Return ooc path

        if predecessor == self.root:
            nodes.reverse()
            return nodes
        nodes.append(predecessor)
        return self.determine_path_to_root(nodes)

    def determine_label_preds_per_lvl(self, labels, preds):
        label_paths = [self.determine_path_to_root([label]) for label in labels]
        pred_paths = [self.determine_path_to_root([pred]) for pred in preds]

        dummy_label = len(self.tree) + 1  # Dummy label used to align length of prediction paths if they differ
        longest_path = max([len(path) for path in label_paths])
        for label_path, pred_path in zip(label_paths, pred_paths):
            # Align length of prediction paths
            if len(label_path) != len(pred_path):
                prediction_difference = len(label_path) - len(pred_path)
                if prediction_difference > 0:
                    while prediction_difference > 0:
                        pred_path.append(dummy_label)
                        prediction_difference -= 1
                else:
                    while prediction_difference < 0:
                        label_path.append(dummy_label)
                        prediction_difference += 1

            #Add dummy to all paths
            while len(label_path) < longest_path and len(pred_path) < longest_path:
                label_path.append(-1)
                pred_path.append(-1)

        #Transpose paths
        label_per_lvl = np.array(label_paths).transpose().tolist()
        preds_per_lvl = np.array(pred_paths).transpose().tolist()

        return label_per_lvl, preds_per_lvl

    #Taken from experiment runner --> Refactor - consolidate functions
    def get_all_nodes_per_lvl(self, level):
        successors = self.tree.successors(self.root)
        while level > 0:
            next_lvl_succesors = []
            for successor in successors:
                next_lvl_succesors.extend(self.tree.successors(successor))
            successors = next_lvl_succesors
            level -= 1
        return successors

    def compute_metrics_transformers_flat(self, pred):
        raw_labels = pred.label_ids
        raw_preds = pred.predictions.argmax(-1)

        labels = [self.transformer_decoder[label]['value'] for label in raw_labels]
        preds = [self.transformer_decoder[pred]['value'] for pred in raw_preds]

        return self.compute_metrics_no_encoding(labels, preds)

    def compute_metrics_transformers_hierarchy(self, pred):
        labels_paths, preds_paths = self.transpose_hierarchy_predictions(pred)

        labels_per_lvl = np.array(labels_paths).transpose().tolist()
        preds_per_lvl = np.array(preds_paths).transpose().tolist()

        labels = [label_path[-1] for label_path in labels_paths]
        preds = [pred_path[-1] for pred_path in preds_paths]

        return self.compute_metrics(labels, preds, labels_per_lvl, preds_per_lvl)

    def transpose_hierarchy_predictions(self, pred):
        labels_paths = pred.label_ids
        preds_paths = []
        for prediction in pred.predictions:
            pred_path = []
            for i in range(len(prediction)):
                # Cut additional zeros!
                if self.num_labels_per_lvl is not None:
                    pred = prediction[i][:self.num_labels_per_lvl[i+1]].argmax(-1)
                else:
                    pred = prediction[i].argmax(-1)
                pred_path.append(pred)
            preds_paths.append(pred_path)

        # Decode hierarchy lvl labels
        for i in range(len(labels_paths[0])):
            nodes = list(self.get_all_nodes_per_lvl(i))
            for label_path in labels_paths:
                if label_path[i] > 0: # Keep 0 (out of category)
                    index = label_path[i] - 1
                    label_path[i] = nodes[index]
            for preds_path in preds_paths:
                if label_path[i] > 0: # Keep 0 (out of category)
                    index = preds_path[i] - 1
                    preds_path[i] = nodes[index]

        return preds_paths, preds_paths

    def compute_metrics_transformers_rnn(self, pred):
        labels, preds, labels_per_lvl, preds_per_lvl = self.transpose_rnn_hierarchy(pred)

        return self.compute_metrics(labels, preds, labels_per_lvl, preds_per_lvl)

    def transpose_rnn_hierarchy(self, pred):
        labels_paths = [list(label) for label in pred.label_ids]
        preds_paths = [list(prediction.argmax(-1)) for prediction in pred.predictions]

        labels_per_lvl = np.array(labels_paths).transpose().tolist()
        preds_per_lvl = np.array(preds_paths).transpose().tolist()

        # Derive leaf predictions
        fill_cat = len(self.tree)

        # Derive prediction
        labels = [self.derive_leaf_node(label_path, fill_cat) for label_path in labels_paths]
        preds = [self.derive_leaf_node(pred_path, fill_cat) for pred_path in preds_paths]

        return labels, preds, labels_per_lvl, preds_per_lvl

    def derive_leaf_node(self, path, fill_cat):
        """Recursively search for the last element that is not the fill up category"""
        last_element = path[-1]
        # Last element found
        if last_element != fill_cat:
            return last_element
        # List becomes empty --> return out of category (0)
        if len(path) == 1:
            return 0

        return self.derive_leaf_node(path[:len(path)-1], fill_cat)

    def compute_metrics_no_encoding(self, labels, preds):
        decoder = dict(self.tree.nodes(data="name"))
        encoder = dict([(value, key) for key, value in decoder.items()])

        # Encoder values to compute metrics
        pp_labels = [encoder[label] for label in labels]
        pp_preds = [encoder[pred] for pred in preds]

        labels_per_lvl, preds_per_lvl = self.determine_label_preds_per_lvl(pp_labels, pp_preds)

        return self.compute_metrics(pp_labels, pp_preds, labels_per_lvl, preds_per_lvl)

    def compute_metrics(self, labels, preds, labels_per_lvl, preds_per_lvl):
        """Compute Metrics for leaf nodes and all nodes in the graph separately"""
        self.logger.debug('Leaf nodes')
        w_prec, w_rec, w_f1, macro_f1 = score_traditional(labels, preds, name=self.experiment_name)  # Score leaf nodes
        h_f_score = hierarchical_score(labels, preds, self.tree, self.root, name=self.experiment_name)

        results = { 'leaf_weighted_prec': w_prec,
                    'leaf_weighted_rec': w_rec,
                    'leaf_weighted_f1': w_f1,
                    'leaf_macro_f1': macro_f1,
                    'h_f1': h_f_score}

        counter = 0

        sum_prec = {'weighted': 0.0, 'macro': 0.0}
        sum_rec = {'weighted': 0.0, 'macro': 0.0}
        sum_f1 = {'weighted': 0.0, 'macro': 0.0}


        for labels_lvl, preds_lvl in zip(labels_per_lvl, preds_per_lvl):

            counter += 1
            labels_lvl = [value for value in labels_lvl if value != -1]
            preds_lvl = [value for value in preds_lvl if value != -1]

            self.logger.debug('lvl_{}'.format(counter))
            score_dict = score_mwpd(labels_lvl, preds_lvl)

            for key in score_dict:
                prec, rec, f1, support = score_dict[key]
                results['{}_prec_lvl_{}'.format(key, counter)] = prec
                results['{}_rec_lvl_{}'.format(key, counter)] = rec
                results['{}_f1_lvl_{}'.format(key, counter)] = f1

                self.logger.info(
                    "{} - Lvl{}: | {}_prec: {:4f} | {}_rec: {:4f} | {}_f1: {:4f}".format(
                        self.experiment_name, counter, key, prec, key, rec, key, f1 ))

                sum_prec[key] += prec
                sum_rec[key] += rec
                sum_f1[key] += f1


        for key in sum_prec:

            avg_prec = sum_prec[key] / len(labels_per_lvl)
            avg_rec = sum_rec[key] / len(labels_per_lvl)
            avg_f1 = sum_f1[key] / len(labels_per_lvl)

            results['average_{}_prec'.format(key)] = avg_prec
            results['average_{}_rec'.format(key)] = avg_rec
            results['average_{}_f1'.format(key)] = avg_f1

            self.logger.info(
                "{} - {} average: | prec: {:4f} | rec: {:4f} | f1: {:4f}".format(
                    self.experiment_name, key, avg_prec, avg_rec, avg_f1))


        return results
