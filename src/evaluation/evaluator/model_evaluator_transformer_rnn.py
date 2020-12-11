import time
import csv
from pathlib import Path
from transformers import RobertaForSequenceClassification, Trainer

from src.evaluation import scorer
from src.evaluation.evaluator.model_evaluator import ModelEvaluator
from src.models.transformers import utils
from src.models.transformers.custom_transformers.roberta_for_hierarchical_classification_rnn import \
    RobertaForHierarchicalClassificationRNN
from src.models.transformers.dataset.category_dataset_flat import CategoryDatasetFlat
from src.utils.result_collector import ResultCollector


class ModelEvaluatorTransformerRNN(ModelEvaluator):

    def __init__(self, configuration_path, test, experiment_type):
        super().__init__(configuration_path, test, experiment_type)

        self.load_model()
        self.load_tree()

    def load_model(self):
        data_dir = Path(self.data_dir)
        file_path = data_dir.joinpath(self.model_path)
        self.model = RobertaForHierarchicalClassificationRNN.from_pretrained(file_path)

    def determine_path_to_root(self, nodes):
        predecessors = self.tree.predecessors(nodes[-1])
        predecessor = [k for k in predecessors][0]

        if predecessor == self.root:
            nodes.reverse()
            return nodes
        nodes.append(predecessor)
        return self.determine_path_to_root(nodes)

    def normalize_path_from_root_per_parent(self, path):
        """Normalize label values per parent node"""
        found_successor = self.root
        normalized_path = []
        for searched_successor in path:
            counter = 0
            successors = self.tree.successors(found_successor)
            for successor in successors:
                counter += 1
                if searched_successor == successor:
                    normalized_path.append(counter)
                    found_successor = searched_successor
                    break

        assert (len(path) == len(normalized_path))
        return normalized_path

    def encode_labels(self):
        """Encode & decode labels plus rescale encoded values"""
        normalized_encoder = {}
        normalized_decoder = {}
        decoder = dict(self.tree.nodes(data="name"))
        encoder = dict([(value, key) for key, value in decoder.items()])

        leaf_nodes = [node[0] for node in self.tree.out_degree(self.tree.nodes()) if node[1] == 0]
        leaf_nodes = [decoder[node] for node in leaf_nodes]

        counter = 0
        for key in encoder:
            if key in leaf_nodes:
                path = self.determine_path_to_root([encoder[key]])
                if 'exploit_hierarchy' in self.parameter and self.parameter['exploit_hierarchy'] == "True":
                    path = self.normalize_path_from_root_per_parent(path)

                normalized_encoder[key] = {'original_key': encoder[key], 'derived_key': counter,
                                           'derived_path': path}
                normalized_decoder[counter] = {'original_key': encoder[key], 'value': key}
                counter += 1

        # Total number of labels is determined by the number of labels in the tree
        number_of_labels = len(self.tree)

        return normalized_encoder, normalized_decoder, number_of_labels

    def evaluate(self):
        ds_eval = self.prepare_eval_dataset()

        normalized_encoder, normalized_decoder, number_leaf_nodes = self.encode_labels()

        evaluator = scorer.HierarchicalScorer(self.experiment_name, self.tree, transformer_decoder=normalized_decoder)
        trainer = Trainer(
            model=self.model,  # the instantiated 🤗 Transformers model to be trained
            compute_metrics=evaluator.compute_metrics_transformers_rnn
        )

        if self.preprocessing:
            texts = list(ds_eval['preprocessed_title'].values)
        else:
            texts = list(ds_eval['title'].values)

        ds_eval['category'] = ds_eval['category'].str.replace(' ', '_')
        labels = list(ds_eval['category'].values)

        tokenizer = utils.provide_tokenizer(self.model_name)

        ds_wdc = CategoryDatasetFlat(texts, labels, tokenizer, normalized_encoder)

        result_collector = ResultCollector(self.dataset_name, self.experiment_type)
        result_collector.results[self.experiment_name] = trainer.evaluate(ds_wdc)

        # Predict values for error analysis
        pred = trainer.predict(ds_wdc)
        labels_paths = pred.label_ids
        preds_paths = []
        for prediction in pred.predictions:
            pred_path = []
            for i in range(len(prediction)):
                # Cut additional zeros!
                pred = prediction[i][:self.num_labels_per_lvl[i+1]].argmax(-1)
                pred_path.append(pred)
            preds_paths.append(pred_path)

        # Decode hierarchy lvl labels
        for i in range(len(labels_paths[0])):
            nodes = list(self.get_all_nodes_per_lvl(i))
            for label_path in labels_paths:
                if label_path[i] > 0: # Keep 0 (out of category)
                    label_path[i] = nodes[label_path[i] - 1]
            for preds_path in preds_paths:
                if label_path[i] > 0: # Keep 0 (out of category)
                    preds_path[i] = nodes[preds_path[i] - 1]

        ds_eval['labels'] = [label_path[-1] for label_path in labels_paths]
        ds_eval['preds'] = [pred_path[-1] for pred_path in preds_paths]

        ds_eval.to_csv(self.prediction_output, index=False, sep=';', encoding='utf-8', quotechar='"',
                                      quoting=csv.QUOTE_ALL)

        # Persist results
        timestamp = time.time()
        result_collector.persist_results(timestamp)
