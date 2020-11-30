import pickle
import time
from pathlib import Path

from src.evaluation import scorer
from src.evaluation.evaluator.model_evaluator import ModelEvaluator
from src.utils.result_collector import ResultCollector


class ModelEvaluatorRandomForest(ModelEvaluator):

    def __init__(self, configuration_path, test, experiment_type):
        super().__init__(configuration_path, test, experiment_type)

        self.load_model()

    def load_model(self):
        data_dir = Path(self.data_dir)
        file_path = data_dir.joinpath(self.model_path)
        self.model = pickle.load(open(file_path, "rb"))

    def evaluate(self):

        result_collector = ResultCollector(self.dataset_name, self.experiment_type)

        ds_eval = self.prepare_eval_dataset()

        # Predict values for error analysis
        ds_eval['prediction'] = self.model.predict(ds_eval['title'])
        y_pred = ds_eval['prediction'].values
        y_true = ds_eval['title'].values

        evaluator = scorer.HierarchicalEvaluator(self.dataset_name, self.experiment_name, None)
        result_collector.results[self.experiment_name] = evaluator.compute_metrics(y_true, y_pred)

        # Persist prediction
        ds_eval.to_pickle(self.prediction_output)
        self.logger.info('Prediction results persisted to {}!'.format(self.prediction_output))

        # Persist results
        timestamp = time.time()
        result_collector.persist_results(timestamp)
