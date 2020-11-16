import time
from pathlib import Path
from transformers import RobertaForSequenceClassification, Trainer

from src.evaluation import evaluation
from src.evaluation.evaluator.model_evaluator import ModelEvaluator
from src.models.transformers import utils
from src.models.transformers.category_dataset import CategoryDataset
from src.utils.result_collector import ResultCollector


class ModelEvaluatorTransformer(ModelEvaluator):

    def __init__(self, configuration_path, test, experiment_type):
        super().__init__(configuration_path, test, experiment_type)

        self.load_model()

    def load_model(self):
        project_dir = Path(__file__).resolve().parents[3]
        file_path = project_dir.joinpath(self.model_path)
        self.model = RobertaForSequenceClassification.from_pretrained(file_path)

    def evaluate(self):
        ds_eval = self.prepare_eval_dataset()

        evaluator = evaluation.HierarchicalEvaluator(self.dataset_name, self.experiment_name, self.encoder)
        trainer = Trainer(
            model=self.model,  # the instantiated 🤗 Transformers model to be trained
            compute_metrics=evaluator.compute_metrics_transformers
        )

        texts = list(ds_eval['title'].values)
        labels = list(ds_eval['category'].values)
        le_dict = dict(zip(self.encoder.classes_, self.encoder.transform(self.encoder.classes_)))

        tokenizer = utils.provide_tokenizer(self.model_name)

        ds_wdc = CategoryDataset(texts, labels, tokenizer, le_dict)

        result_collector = ResultCollector(self.dataset_name, self.experiment_type)
        result_collector.results[self.experiment_name] = trainer.evaluate(ds_wdc)

        # Predict values for error analysis
        prediction = trainer.predict(ds_wdc)
        preds = prediction.predictions.argmax(-1)
        ds_eval['prediction'] = self.encoder.inverse_transform(preds)

        ds_eval.to_pickle(self.prediction_output)

        # Persist results
        timestamp = time.time()
        result_collector.persist_results(timestamp)
