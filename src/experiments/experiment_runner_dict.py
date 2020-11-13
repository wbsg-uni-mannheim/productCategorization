import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.evaluation import evaluation
from src.experiments.experiment_runner import ExperimentRunner
from src.models.dictionary.dictclassifier import DictClassifier

from src.utils.result_collector import ResultCollector


class ExperimentRunnerDict(ExperimentRunner):

    def __init__(self, path, test, experiment_type):
        super().__init__(path, test, experiment_type)

        self.most_frequent_leaf = None

        self.load_experiments(path)
        self.load_datasets()

    def load_experiments(self, path):
        """Load experiments defined in the json for which a path is provided"""
        experiments = self.load_configuration(path)
        self.most_frequent_leaf = experiments['most_frequent_leaf']

        # Normalise experiment parameter
        for parameters in experiments['parameter']:
            for parameter, value in parameters.items():
                if value == 'True':
                    parameters[parameter] = True
                elif value == 'False':
                    parameters[parameter] = False

        self.parameter = experiments['parameter']

    def run(self):
        """Run experiments"""
        result_collector = ResultCollector(self.dataset_name, self.experiment_type)

        dict_classifier = DictClassifier(self.dataset_name, self.most_frequent_leaf)

        # fallback classifier
        pipeline = Pipeline([
            ('vect', CountVectorizer()),
            ('clf', MultinomialNB()),
        ])

        classifier_dictionary_based = pipeline.fit(self.dataset['train']['title'].values,
                                                   self.dataset['train']['category'].values)

        for configuration in self.parameter:
            y_true = self.dataset['validate']['category'].values

            fallback_classifier = None
            if configuration['fallback']:
                fallback_classifier = classifier_dictionary_based

            y_pred = dict_classifier.classify_dictionary_based(self.dataset['validate']['title'],
                                                               fallback_classifier, configuration['lemmatizing'],
                                                               configuration['synonyms'])

            experiment_name = '{}; title only; synonyms: {}, lemmatizing: {}, fallback: {}'.format(
                self.experiment_type, configuration['synonyms'],
                configuration['lemmatizing'], configuration['fallback'])

            evaluator = evaluation.HierarchicalEvaluator(self.dataset_name, experiment_name, None)
            result_collector.results[experiment_name] = evaluator.compute_metrics(y_true, y_pred)

        # Persist results
        timestamp = time.time()
        result_collector.persist_results(timestamp)
