import pickle
import time

import fasttext
import csv
import os

from src.data.preprocessing import preprocess
from src.evaluation import scorer
from src.experiments.runner.experiment_runner import ExperimentRunner
from src.utils.result_collector import ResultCollector



class ExperimentRunnerFastText(ExperimentRunner):

    def __init__(self, path, test, experiment_type):
        super().__init__(path, test, experiment_type)

        self.load_experiments(path)
        self.load_datasets()
        self.fasttextencoder = {}
        
        self.load_tree()

    def load_experiments(self, path):
        """Load experiments defined in the json for which a path is provided"""
        experiments = self.load_configuration(path)
        self.parameter = experiments['parameter']

    def prepare_fasttext(self, ds, split):
        ds['category'] = ds['category'].str.replace(' ', '_')
        ds['category_prepared'] = '__label__' + ds['category'].astype(str)

        #Preprocess Title
        ds['title'] =ds['title'].apply(preprocess)

        orig_categories = ds['category'].values
        prepared_categories = ds['category_prepared'].values

        prep_keys = dict(set(zip(prepared_categories, orig_categories)))
        self.fasttextencoder = {**self.fasttextencoder, **prep_keys}

        #Use only title for prediction
        ds = ds[['title', 'category_prepared']]

        #Save prepared ds to disk
        path = '{}/data/processed/{}/fasttext/{}-{}.csv'.format(self.data_dir, self.dataset_name, self.parameter['experiment_name'], split)
        ds.to_csv(path, index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, escapechar=" ")

        return path, ds, orig_categories

    def run(self):
        """Run experiments"""
        result_collector = ResultCollector(self.dataset_name, self.experiment_type)

        # Reduce data if run in test mode:
        if self.test:
            for key in self.dataset:
                self.dataset[key] = self.dataset[key][:50]
            self.logger.warning('Run in test mode - dataset reduced to 50 records!')

        # Train Classifier on train and validate
        ds_train = self.dataset['train']
        ds_validate = self.dataset['validate']
        ds_test = self.dataset['test']

        #Prepare data
        train_path, ds_train, orig_categories_train = self.prepare_fasttext(ds_train, 'train')
        validate_path, ds_validate, orig_categories_validate = self.prepare_fasttext(ds_validate, 'validate')
        test_path, ds_test, orig_categories_test  = self.prepare_fasttext(ds_test, 'test')

        y_true = list(orig_categories_validate)

        # Use best performing configuration according to Nils' results! - Run more experiments if necessary
        if self.parameter['autotune'] == "True":
            classifier = fasttext.train_supervised(input=train_path, autotuneValidationFile=validate_path,
                                                   autotuneMetric="f1")
        else:
            classifier = fasttext.train_supervised(input=train_path, epoch=self.parameter['epoch'],
                                                   wordNgrams=self.parameter['wordNgrams'],
                                                   loss=self.parameter['loss'], minn=self.parameter['minn'],
                                                   maxn=self.parameter['maxn'], neg=self.parameter['neg'],
                                                   thread=self.parameter['thread'], dim=self.parameter['dim']
            )

        y_pred, y_prob = classifier.predict(ds_validate['title'].values.tolist())
        # Postprocess labels
        y_pred = [self.fasttextencoder[prediction[0]] for prediction in y_pred]

        evaluator = scorer.HierarchicalScorer(self.parameter['experiment_name'], self.tree)
        result_collector.results[self.parameter['experiment_name']] = evaluator.compute_metrics_no_encoding(y_true, y_pred)


        # Save classifier
        output_file = '{}/models/{}/fasttext/model/{}.bin'\
            .format(self.data_dir, self.dataset_name, self.parameter['experiment_name'])
        classifier.save_model(output_file)
        self.logger.info('Classifier serialized to file {}'.format(output_file))

        # Save classifier
        output_file = '{}/models/{}/fasttext/model/encoder-{}.pkl' \
            .format(self.data_dir, self.dataset_name, self.parameter['experiment_name'])
        with open(output_file, "wb") as file:
            pickle.dump(self.fasttextencoder, file=file)
        self.logger.info('Encoder serialized to file {}'.format(output_file))

        # Persist results
        timestamp = time.time()
        result_collector.persist_results(timestamp)
