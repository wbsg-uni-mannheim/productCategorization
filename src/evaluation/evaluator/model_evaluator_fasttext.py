import csv
import pickle
import time
from pathlib import Path

import fasttext

from src.data.preprocessing import preprocess
from src.evaluation import scorer
from src.evaluation.evaluator.model_evaluator import ModelEvaluator
from src.utils.result_collector import ResultCollector


class ModelEvaluatorFastText(ModelEvaluator):

    def __init__(self, configuration_path, test, experiment_type):
        super().__init__(configuration_path, test, experiment_type)

        self.load_model()
        self.encoder = None

        self.initialize_encoder()

    def load_model(self):

        self.model = fasttext.load_model(self.model_path)

    def initialize_encoder(self):
        project_dir = Path(__file__).resolve().parents[3]
        file_path = project_dir.joinpath(self.parameter['encoder_path'])
        with open(file_path, 'rb') as encoder_file:
            self.encoder = pickle.load(encoder_file)

    def prepare_fasttext(self, ds, split):
        ds['category_prepared'] = ds['category'].str.replace(' ', '_')
        ds['category_prepared'] = '__label__' + ds['category_prepared'].astype(str)

        #Preprocess Title
        ds['title'] =ds['title'].apply(preprocess)

        orig_categories = ds['category'].values
        prepared_categories = ds['category_prepared'].values

        #Use only title for prediction
        ds = ds[['title', 'category_prepared']]

        #Save prepared ds to disk
        path = './data/processed/{}/fasttext/{}-{}.csv'.format(self.dataset_name, self.experiment_name, split)
        ds.to_csv(path, index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, escapechar=" ")

        return path, ds

    def evaluate(self):

        result_collector = ResultCollector(self.dataset_name, self.experiment_type)

        ds_eval = self.prepare_eval_dataset()
        y_true = ds_eval['category'].values

        #Preprocess data
        eval_path, ds_eval = self.prepare_fasttext(ds_eval, 'eval')

        y_pred, y_prob = self.model.predict(ds_eval['title'].values.tolist())
        # Postprocess labels
        y_pred = [self.encoder[prediction[0]] for prediction in y_pred]

        normalized_encoder, normalized_decoder, number_leaf_nodes = self.encode_labels()

        evaluator = scorer.HierarchicalScorer(self.experiment_name, self.tree, transformer_decoder=normalized_decoder)

        result_collector.results[self.experiment_name] = evaluator.compute_metrics_transformers_flat(y_true, y_pred)

        # Persist prediction
        ds_eval.to_pickle(self.prediction_output)
        self.logger.info('Prediction results persisted to {}!'.format(self.prediction_output))

        # Persist results
        timestamp = time.time()
        result_collector.persist_results(timestamp)
