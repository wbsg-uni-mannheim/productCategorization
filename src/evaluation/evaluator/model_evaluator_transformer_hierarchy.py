import time
import csv
from pathlib import Path
from transformers import Trainer

from src.evaluation import scorer
from src.evaluation.evaluator.model_evaluator import ModelEvaluator
from src.models.transformers import utils
from src.models.transformers.custom_transformers.roberta_for_hierarchical_classification_hierarchy import \
    RobertaForHierarchicalClassificationHierarchy
from src.models.transformers.custom_transformers.roberta_for_hierarchical_classification_rnn import \
    RobertaForHierarchicalClassificationRNN
from src.models.transformers.dataset.category_dataset_flat import CategoryDatasetFlat
from src.models.transformers.dataset.category_dataset_rnn import CategoryDatasetRNN
from src.utils.result_collector import ResultCollector


class ModelEvaluatorTransformerHierarchy(ModelEvaluator):

    def __init__(self, configuration_path, test, experiment_type):
        super().__init__(configuration_path, test, experiment_type)

        self.load_model()
        self.load_tree()

    def load_model(self):
        data_dir = Path(self.data_dir)
        file_path = data_dir.joinpath(self.model_path)
        print(file_path)
        self.model = RobertaForHierarchicalClassificationHierarchy.from_pretrained(file_path)

    def intialize_hierarchy_paths(self):
        """initialize paths using the provided tree"""

        leaf_nodes = [node[0] for node in self.tree.out_degree if node[1] == 0]
        paths = [self.tree_utils.determine_path_to_root([node]) for node in leaf_nodes]

        # Normalize paths per level in hierarchy - currently the nodes are of increasing number throughout the tree.
        normalized_paths = [self.tree_utils.normalize_path_from_root_per_level(path) for path in paths]

        normalized_encoder = {'Root': {'original_key': 0, 'derived_key': 0}}
        normalized_decoder = { 0: {'original_key': 0, 'value': 'Root'}}
        decoder = dict(self.tree.nodes(data="name"))
        encoder = dict([(value, key) for key, value in decoder.items()])

        #initiaize encoders
        for path, normalized_path in zip(paths, normalized_paths):
            key = path[-1]
            derived_key = normalized_path[-1]
            if key in leaf_nodes:
                normalized_encoder[decoder[key]] = {'original_key': key, 'derived_key': derived_key}
                normalized_decoder[derived_key] = {'original_key': key, 'value': decoder[key]}

        oov_path = [[0, 0, 0]]
        normalized_paths = oov_path + normalized_paths

        #Align length of paths if necessary
        longest_path = max([len(path) for path in normalized_paths])

        # Sort paths ascending
        sorted_normalized_paths = []
        for i in range(len(normalized_paths)):
            found_path = normalized_paths[0]
            for path in normalized_paths:
                for found_node, node in zip(found_path,path):
                    if found_node > node:
                        found_path = path
                        break

            if not (found_path is None):
                sorted_normalized_paths.append(found_path)
                normalized_paths.remove(found_path)

        return normalized_encoder, normalized_decoder, sorted_normalized_paths

    def evaluate(self):
        ds_eval = self.prepare_eval_dataset()

        normalized_encoder, normalized_decoder, number_of_labels = self.intialize_hierarchy_paths()

        evaluator = scorer.HierarchicalScorer(self.experiment_name, self.tree, transformer_decoder=normalized_decoder)
        trainer = Trainer(
            model=self.model,  # the instantiated 🤗 Transformers model to be trained
            compute_metrics=evaluator.compute_metrics_transformers_flat
        )

        if self.preprocessing:
            texts = list(ds_eval['preprocessed_title'].values)
        else:
            texts = list(ds_eval['title'].values)

        ds_eval['category'] = ds_eval['category'].str.replace(' ', '_')
        labels = list(ds_eval['category'].values)

        tokenizer = utils.roberta_base_tokenizer()

        ds_wdc = CategoryDatasetFlat(texts, labels, tokenizer, normalized_encoder)

        result_collector = ResultCollector(self.dataset_name, self.experiment_type)
        result_collector.results[self.experiment_name] = trainer.evaluate(ds_wdc)

        # Predict values for error analysis
        prediction = trainer.predict(ds_wdc)
        preds = prediction.predictions.argmax(-1)
        ds_eval['prediction'] = [normalized_decoder[pred]['value'] for pred in preds]
        full_prediction_output = '{}/{}'.format(self.data_dir, self.prediction_output)

        ds_eval.to_csv(full_prediction_output, index=False, sep=';', encoding='utf-8', quotechar='"',
                       quoting=csv.QUOTE_ALL)

        # Persist results
        timestamp = time.time()
        result_collector.persist_results(timestamp)