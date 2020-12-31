import time
import csv
from pathlib import Path
from transformers import Trainer

from src.evaluation import scorer
from src.evaluation.evaluator.model_evaluator import ModelEvaluator
from src.models.transformers import utils
from src.models.transformers.custom_transformers.roberta_for_hierarchical_classification_rnn import \
    RobertaForHierarchicalClassificationRNN
from src.models.transformers.dataset.category_dataset_rnn import CategoryDatasetRNN
from src.utils.result_collector import ResultCollector


class ModelEvaluatorTransformerRNN(ModelEvaluator):

    def __init__(self, configuration_path, test, experiment_type):
        super().__init__(configuration_path, test, experiment_type)

        self.load_model()
        self.load_tree()

    def load_model(self):
        data_dir = Path(self.data_dir)
        file_path = data_dir.joinpath(self.model_path)
        print(file_path)
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
        longest_path = 0
        for key in encoder:
            if key in leaf_nodes:
                path = self.determine_path_to_root([encoder[key]])
                if 'exploit_hierarchy' in self.parameter and self.parameter['exploit_hierarchy']:
                    path = self.normalize_path_from_root_per_parent(path)

                normalized_encoder[key] = {'original_key': encoder[key], 'derived_key': counter,
                                           'derived_path': path}
                normalized_decoder[counter] = {'original_key': encoder[key], 'value': key}
                if len(path) > longest_path:
                    longest_path = len(path)

                counter += 1

        # Align path length
        fill_up_category = len(self.tree)

        for key in normalized_encoder:
            while len(normalized_encoder[key]['derived_path']) < longest_path:
                normalized_encoder[key]['derived_path'].append(fill_up_category)

        # Total number of labels is determined by the number of labels in the tree + 1 for out of category
        number_of_labels = len(self.tree) + 1

        return normalized_encoder, normalized_decoder, number_of_labels

    def evaluate(self):
        ds_eval = self.prepare_eval_dataset()

        normalized_encoder, normalized_decoder, number_of_labels = self.encode_labels()

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

        tokenizer = utils.roberta_base_tokenizer()

        ds_wdc = CategoryDatasetRNN(texts, labels, tokenizer, normalized_encoder)

        result_collector = ResultCollector(self.dataset_name, self.experiment_type)
        result_collector.results[self.experiment_name] = trainer.evaluate(ds_wdc)

        # Predict values for error analysis
        pred = trainer.predict(ds_wdc)

        labels, preds, labels_per_lvl, preds_per_lvl = evaluator.transpose_rnn_hierarchy(pred)

        ds_eval['Leaf Label'] = [normalized_decoder[label] for label in labels]
        ds_eval['Leaf Prediction'] = [normalized_decoder[pred] for pred in preds]

        counter = 1
        for labs, predictions in zip(labels_per_lvl, preds_per_lvl):
            column_name_label = 'Hierarchy Level {} Label'.format(counter)
            column_name_prediction = 'Hierarchy Level {} Prediction'.format(counter)

            ds_eval[column_name_label] = [normalized_decoder[label] for label in labs]
            ds_eval[column_name_prediction] = [normalized_decoder[prediction] for prediction in predictions]

            counter += 1



        ds_eval.to_csv(self.prediction_output, index=False, sep=';', encoding='utf-8', quotechar='"',
                                      quoting=csv.QUOTE_ALL)

        # Persist results
        timestamp = time.time()
        result_collector.persist_results(timestamp)
