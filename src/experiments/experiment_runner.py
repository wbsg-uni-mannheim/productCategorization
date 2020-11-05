import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.evaluation import evaluation
from src.models.transformers import utils
from src.models.transformers.category_dataset import CategoryDataset
from src.models.dictionary.dictclassifier import DictClassifier
import time

from transformers import TrainingArguments, Trainer

from src.utils.result_collector import ResultCollector


class ExperimentRunner:

    def __init__(self, path):
        self.logger = logging.getLogger(__name__)

        self.path = path
        self.experiment_type = None
        self.dataset_name = None
        self.dataset = {}
        self.parameter = None
        self.most_frequent_leaf = None
        self.evaluate_wdc = None

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

        if self.experiment_type == 'dict-based':
            self.parameter = experiments['parameter']
        else:
            self.parameter = experiments['parameter'][0]

    def load_datasets(self, dataset_name):
        """Load dataset for the given experiments"""
        project_dir = Path(__file__).resolve().parents[2]
        splits = ['train', 'validate', 'test']

        for split in splits:
            relative_path = 'data/processed/{}/split/raw/{}_data_{}.pkl'.format(dataset_name, split, dataset_name)
            file_path = project_dir.joinpath(relative_path)
            self.dataset[split] = pd.read_pickle(file_path)

        self.logger.info('Loaded dataset {}!'.format(dataset_name))

    def run(self):
        """Run experiments"""
        result_collector = ResultCollector(self.dataset_name, self.experiment_type)

        if self.experiment_type == 'dict-based':
            dict_classifier = DictClassifier(self.dataset_name, self.most_frequent_leaf)

            # fallback classifier
            pipeline = Pipeline([
                ('vect', CountVectorizer()),
                ('clf', MultinomialNB()),
            ])
            classifier_dictionary_based = pipeline.fit(self.dataset['train']['title'].values,
                                                       self.dataset['train']['category'].values)

            for configuration in self.parameter:
                y_true = self.dataset['validate']['category'].values[:20]

                fallback_classifier = None
                if configuration['fallback']:
                    fallback_classifier = classifier_dictionary_based

                y_pred = dict_classifier.classify_dictionary_based(self.dataset['validate']['title'][:20],
                                                                   fallback_classifier, configuration['lemmatizing'],
                                                                   configuration['synonyms'])

                print(y_true)
                print('____________')
                print(y_pred)
                experiment_name = '{}; title only; synonyms: {}, lemmatizing: {}, fallback: {}'.format(
                    self.experiment_type, configuration['synonyms'],
                    configuration['lemmatizing'], configuration['fallback'])

                evaluator = evaluation.HierarchicalEvaluator(self.dataset_name, experiment_name, None)
                result_collector.results[experiment_name] = evaluator.compute_metrics(y_true, y_pred)

        elif self.experiment_type == 'transformer-based':
            encoder = LabelEncoder()
            encoder.fit(self.dataset['train']['category'].values)
            le_dict = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

            tokenizer, model = utils.provide_model_and_tokenizer(self.parameter['model_name'], len(le_dict) + 1)

            tf_ds = {}
            for key in self.dataset:
                df_ds = self.dataset[key][:10]
                texts = list(df_ds['title'].values)
                labels = list(df_ds['category'].values)

                tf_ds[key] = CategoryDataset(texts, labels, tokenizer, le_dict)

            training_args = TrainingArguments(
                    output_dir='./experiments/{}/transformers/results/model/{}'
                        .format(self.dataset_name, self.parameter['experiment_name']),
                    # output directory
                    num_train_epochs=self.parameter['epochs'],  # total # of training epochs
                    learning_rate=self.parameter['learning_rate'],
                    per_device_train_batch_size=self.parameter['per_device_train_batch_size'],  # batch size per device during training
                    per_device_eval_batch_size=64,  # batch size for evaluation
                    warmup_steps=500,  # number of warmup steps for learning rate scheduler
                    weight_decay=self.parameter['weight_decay'],  # strength of weight decay
                    logging_dir='./experiments/{}/transformers/logs'.format(self.dataset_name),
                    # directory for storing logs
                    save_total_limit=5,  # Save only the last 5 Checkpoints
                    metric_for_best_model=self.parameter['metric_for_best_model'],
                    load_best_model_at_end=True,
                    gradient_accumulation_steps=2,
                    seed=self.parameter['seed']
            )

            evaluator = evaluation.HierarchicalEvaluator(self.dataset_name, self.parameter['experiment_name'], encoder)
            trainer = Trainer(
                model=model,  # the instantiated 🤗 Transformers model to be trained
                    args=training_args,  # training arguments, defined above
                    train_dataset=tf_ds['train'],  # tensorflow_datasets training dataset
                    eval_dataset=tf_ds['validate'],  # tensorflow_datasets evaluation dataset
                    compute_metrics=evaluator.compute_metrics_transformers
                )

            trainer.train()
            result_collector.results['{}-{}'.format(self.parameter['experiment_name'], 'train')] \
                = trainer.evaluate(tf_ds['train'])
            result_collector.results['{}-{}'.format(self.parameter['experiment_name'], 'validate')] \
                = trainer.evaluate(tf_ds['validate'])
            result_collector.results['{}-{}'.format(self.parameter['experiment_name'], 'test')] \
                = trainer.evaluate(tf_ds['test'])
            trainer.save_model()

            # Persist results
            timestamp = time.time()
            result_collector.persist_results(timestamp)
