import json
import logging
from pathlib import Path
import os
import csv

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.evaluation import evaluation
from src.models.dictionary.dictclassifier import DictClassifier
from datetime import datetime
import time


class ExperimentRunner:

    def __init__(self, path):
        self.logger = logging.getLogger(__name__)

        self.path = path
        self.experiment_type = None
        self.dataset_name = None
        self.dataset = {}
        self.parameter = None
        self.most_frequent_leaf = None

        self.results = None

        self.load_experiments(path)
        self.load_datasets(self.dataset_name)

    def __str__(self):
        output = 'Experiment runner for {} experiments on {} dataset with the following parameter: {}' \
            .format(self.experiment_type, self.dataset_name, self.parameter)
        return output

    def load_experiments(self, path):
        """Load experiments defined in the json for which a path is provided"""
        with open(path) as json_file:
            experiments = json.load(json_file)
            self.logger.info('Loaded experiments from {}!'.format(path))

        self.dataset_name = experiments['dataset']
        self.experiment_type = experiments['type']
        if self.experiment_type == 'dict-based':
            self.most_frequent_leaf = experiments['most_frequent_leaf']

        # Normalise experiment parameter
        for parameters in experiments['parameter']:
            for parameter, value in parameters.items():
                if value == 'True':
                    parameters[parameter] = True
                elif value == 'False':
                    parameters[parameter] = False
        self.parameter = experiments['parameter']

    def load_datasets(self, dataset_name):
        """Load dataset for the given experiments"""
        project_dir = Path(__file__).resolve().parents[2]
        splits = ['train', 'validate', 'test']

        for split in splits:
            relative_path = 'data/processed/{}/split/raw/{}_data_{}.pkl'.format(dataset_name, split, dataset_name)
            file_path = project_dir.joinpath(relative_path)
            self.dataset[split] = pd.read_pickle(file_path)

        self.logger.info('Loaded dataset {}!'.format(dataset_name))

    def persist_results(self, results, timestamp):
        project_dir = Path(__file__).resolve().parents[2]
        relative_path = 'experiments/{}/results/'.format(self.dataset_name)
        absolute_path = project_dir.joinpath(relative_path)

        if not os.path.exists(absolute_path):
            os.mkdir(absolute_path)

        file_path = absolute_path.joinpath('{}_{}_results_{}.csv'.format(
                            self.dataset_name, self.experiment_type,
                            datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d_%H-%M-%S')))
        header = ['Experiment Name', 'weighted_prec', 'weighted_recall', 'weighted_f1', 'macro_f1', 'h_f1']

        rows = []
        for result in results.keys():
            row = [result, results[result]['weighted_prec'],
                   results[result]['weighted_rec'], results[result]['weighted_f1'],
                   results[result]['macro_f1'], results[result]['h_f1']]
            rows.append(row)

        # Write to csv
        with open(file_path, 'w', newline='') as f:
            csv_writer = csv.writer(f, delimiter=';')

            csv_writer.writerow(header)
            csv_writer.writerows(rows)

        self.logger.info('Results of {} on {} written to file {}!'.format(
                            self.experiment_type, self.dataset_name, file_path.absolute()))

    def run(self):
        """Run experiments"""
        if self.experiment_type == 'dict-based':
            dict_classifier = DictClassifier(self.dataset_name, self.most_frequent_leaf)

            # fallback classifier
            pipeline = Pipeline([
                ('vect', CountVectorizer()),
                ('clf', MultinomialNB()),
            ])

            classifier_dictionary_based = pipeline.fit(self.dataset['train']['title'].values,
                                                       self.dataset['train']['category'].values)
            root = [node[0] for node in dict_classifier.tree.in_degree if node[1] == 0][0]

            result_collector = {}

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

                scores_all_labels = evaluation.eval_traditional(experiment_name, y_true, y_pred)
                h_f_score = evaluation.hierarchical_eval(experiment_name, y_true, y_pred, dict_classifier.tree, root)

                # To-Do: Change structure of elements!
                result_collector[experiment_name] = {'weighted_prec': scores_all_labels[0][0][0],
                                                     'weighted_rec': scores_all_labels[0][0][1],
                                                     'weighted_f1': scores_all_labels[0][0][2],
                                                     'macro_f1': scores_all_labels[0][1],
                                                     'h_f1': h_f_score}
            timestamp = time.time()
            self.persist_results(result_collector, timestamp)
