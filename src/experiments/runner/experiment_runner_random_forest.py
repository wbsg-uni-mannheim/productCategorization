import time
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline


from src.data.preprocessing import preprocess
from src.evaluation import evaluation
from src.experiments.runner.experiment_runner import ExperimentRunner
from src.utils.result_collector import ResultCollector



class ExperimentRunnerRandomForest(ExperimentRunner):

    def __init__(self, path, test, experiment_type):
        super().__init__(path, test, experiment_type)

        self.load_experiments(path)
        self.load_datasets()

    def load_experiments(self, path):
        """Load experiments defined in the json for which a path is provided"""
        experiments = self.load_configuration(path)
        self.parameter = experiments['parameter']

    def run(self):
        """Run experiments"""
        result_collector = ResultCollector(self.dataset_name, self.experiment_type)

        pipeline = Pipeline([
            ('vect', CountVectorizer(preprocessor=preprocess)),
            ('clf', RandomForestClassifier()),
        ])

        # Reduce data if run in test mode:
        if self.test:
            for key in self.dataset:
                self.dataset[key] = self.dataset[key][:50]
            self.logger.warning('Run in test mode - dataset reduced to 50 records!')

        # Train Classifier on train and validate
        ds_train = self.dataset['train'].append(self.dataset['validate'])
        ds_test = self.dataset['test']

        classifier = pipeline.fit(ds_train['title'], ds_train['category'])

        y_pred = classifier.predict(ds_test['title'])
        y_true = ds_test['category'].values

        evaluator = evaluation.HierarchicalEvaluator(self.dataset_name, self.parameter['experiment_name'], None)
        result_collector.results[self.parameter['experiment_name']] = evaluator.compute_metrics(y_true, y_pred)


        # Save classifier
        output_file = './experiments/{}/random_forest/model/{}.pkl'\
            .format(self.dataset_name, self.parameter['experiment_name'])
        with open(output_file, "wb") as file:
            pickle.dump(classifier, file=file)

        self.logger.info('Classifier serialized to file {}'.format(output_file))
        # Persist results
        timestamp = time.time()
        result_collector.persist_results(timestamp)
