import time
from pathlib import Path
from transformers import RobertaForSequenceClassification, Trainer

from src.evaluation import scorer
from src.evaluation.evaluator.model_evaluator import ModelEvaluator
from src.models.transformers import utils
from src.models.transformers.dataset.category_dataset_flat import CategoryDatasetFlat
from src.utils.result_collector import ResultCollector


class ModelEvaluatorTransformer(ModelEvaluator):

    def __init__(self, configuration_path, test, experiment_type):
        super().__init__(configuration_path, test, experiment_type)

        self.load_model()
        self.load_tree()

    def load_model(self):
        data_dir = Path(self.data_dir)
        file_path = data_dir.joinpath(self.model_path)
        self.model = RobertaForSequenceClassification.from_pretrained(file_path)

    def encode_labels(self):
        """Encode & decode labels plus rescale encoded values"""
        normalized_encoder = {}
        normalized_decoder = {}
        decoder = dict(self.tree.nodes(data="name"))
        encoder = dict([(value, key) for key, value in decoder.items()])

        leaf_nodes = [node[0] for node in self.tree.out_degree(self.tree.nodes()) if node[1] == 0]
        leaf_nodes = [decoder[node] for node in leaf_nodes]
        number_leaf_nodes = len(leaf_nodes) + 1

        # Rescale keys!
        derived_key = 1 # Start with 1 --> 0 is out of category
        for key in encoder:
            if key in leaf_nodes:
                normalized_encoder[key] = {'original_key': encoder[key], 'derived_key': derived_key}
                normalized_decoder[derived_key] = {'original_key': encoder[key], 'value': key}
                derived_key += 1

        return normalized_encoder, normalized_decoder, number_leaf_nodes

    def evaluate(self):
        ds_eval = self.prepare_eval_dataset()

        normalized_encoder, normalized_decoder, number_leaf_nodes = self.encode_labels()

        evaluator = scorer.HierarchicalScorer(self.experiment_name, self.tree, transformer_decoder=normalized_decoder)
        trainer = Trainer(
            model=self.model,  # the instantiated 🤗 Transformers model to be trained
            compute_metrics=evaluator.compute_metrics_transformers_flat
        )

        if self.preprocessing:
            texts = list(ds_eval['preprocessed_title'].values)
        else:
            texts = list(ds_eval['title'].values)
            
        labels = [value.replace(' ', '_') for value in ds_eval['category'].values]

        tokenizer = utils.provide_tokenizer(self.model_name)

        ds_wdc = CategoryDatasetFlat(texts, labels, tokenizer, normalized_encoder)

        result_collector = ResultCollector(self.dataset_name, self.experiment_type)
        result_collector.results[self.experiment_name] = trainer.evaluate(ds_wdc)

        # Predict values for error analysis
        prediction = trainer.predict(ds_wdc)
        preds = prediction.predictions.argmax(-1)
        ds_eval['prediction'] = [normalized_decoder[pred]['value'] for pred in preds]

        ds_eval.to_pickle(self.prediction_output)

        # Persist results
        timestamp = time.time()
        result_collector.persist_results(timestamp)
