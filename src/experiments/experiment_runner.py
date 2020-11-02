import json
import logging
from pathlib import Path
import os
import csv

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.evaluation import evaluation
from src.models.bert import utils
from src.models.bert.category_dataset import CategoryDataset
from src.models.dictionary.dictclassifier import DictClassifier
from datetime import datetime
import time

from transformers import TrainingArguments, Trainer


class ExperimentRunner:

    def __init__(self, path):
        self.logger = logging.getLogger(__name__)

        self.path = path
        self.experiment_type = None
        self.dataset_name = None
        self.dataset = {}
        self.wdc = None
        self.parameter = None
        self.most_frequent_leaf = None

        self.results = None

        self.load_experiments(path)
        #Evaluate on WDC - find smarter solution(!)
        self.load_datasets('webdatacommons')
        self.wdc = self.dataset

        full_dataset = pd.DataFrame()

        for key in self.dataset:
            self.dataset[key]['dataset'] = key
            self.wdc = full_dataset.append(self.dataset[key])

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
            if self.experiment_type == 'dict-based':
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
        """Persist Experiment Results"""
        project_dir = Path(__file__).resolve().parents[2]
        relative_path = 'experiments/{}/results/'.format(self.dataset_name)
        absolute_path = project_dir.joinpath(relative_path)

        if not os.path.exists(absolute_path):
            os.mkdir(absolute_path)

        file_path = absolute_path.joinpath('{}_{}_results_{}.csv'.format(
                            self.dataset_name, self.experiment_type,
                            datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d_%H-%M-%S')))

        header = ['Experiment Name','Dataset']
        # Use first experiment as reference for the metric header
        metric_header = list(list(results.values())[0].keys())
        header = header + metric_header

        rows = []
        for result in results.keys():
            row = [result, self.dataset_name]
            for metric in results[result].items():
                row.append(metric[1])
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

                eval = evaluation.TransformersEvaluator(self.dataset_name, experiment_name)
                result_collector[experiment_name] = eval.compute_metrics(y_true, y_pred)

            timestamp = time.time()
            self.persist_results(result_collector, timestamp)

        elif self.experiment_type == 'transformer-based':
            for parameter in self.parameter:
                encoder = LabelEncoder()
                encoder.fit(self.dataset['train']['category'].values)
                le_dict = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

                tokenizer, model = utils.provide_model_and_tokenizer(parameter['model_name'], len(le_dict)+1)

                tf_ds = {}
                for key in self.dataset:
                    df_ds = self.dataset[key]
                    texts = list(df_ds['title'].values)
                    labels = list(df_ds['category'].values)

                    tf_ds[key] = CategoryDataset(texts, labels, tokenizer, le_dict)

                training_args = TrainingArguments(
                output_dir='./experiments/{}/bert/results'.format(self.dataset_name),  # output directory
                num_train_epochs=3,  # total # of training epochs
                per_device_train_batch_size=16,  # batch size per device during training
                per_device_eval_batch_size=64,  # batch size for evaluation
                warmup_steps=500,  # number of warmup steps for learning rate scheduler
                weight_decay=0.01,  # strength of weight decay
                logging_dir='./experiments/{}/bert/logs'.format(self.dataset_name),  # directory for storing logs
                save_total_limit=5 # Save only the last 5 Checkpoints
                )

                eval = evaluation.TransformersEvaluator(self.dataset_name, parameter['experiment_name'])
                trainer = Trainer(
                model=model,  # the instantiated 🤗 Transformers model to be trained
                args=training_args,  # training arguments, defined above
                train_dataset=tf_ds['train'],  # tensorflow_datasets training dataset
                eval_dataset=tf_ds['validate'],  # tensorflow_datasets evaluation dataset
                compute_metrics=eval.compute_metrics_transformers
                )

                trainer.train()
                result_collector = {}
                result_collector[parameter['experiment_name']] = trainer.evaluate(tf_ds['test'])
                result_collector['{}-wdc'.format(parameter['experiment_name'])] = trainer.evaluate(self.wdc)
                trainer.save_model()

                timestamp = time.time()
                self.persist_results(result_collector, timestamp)

