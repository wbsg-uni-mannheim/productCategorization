import json
import logging
import time
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import RobertaForSequenceClassification, TrainingArguments, Trainer

from src.evaluation import evaluation
from src.models.transformers import utils
from src.models.transformers.category_dataset import CategoryDataset
from src.utils.result_collector import ResultCollector


class ModelEvaluator():

    def __init__(self, configuration_path):
        self.logger = logging.getLogger(__name__)
        self.dataset_name = None
        self.dataset = {}
        self.full_dataset = None
        self.model = None
        self.encoder = None

        self.load_configuration(configuration_path)
        self.load_datasets()
        self.initialize_encoder() # Preprocess dataset labels as done for training dataset

        project_dir = Path(__file__).resolve().parents[2]
        file_path = project_dir.joinpath(self.model_path)
        self.model = RobertaForSequenceClassification.from_pretrained(file_path)

    def load_datasets(self):
        """Load dataset for the given experiments"""
        project_dir = Path(__file__).resolve().parents[2]
        splits = ['train', 'validate', 'test']

        self.full_dataset = pd.DataFrame()

        for split in splits:
            relative_path = 'data/processed/{}/split/raw/{}_data_{}.pkl'\
                                .format(self.dataset_name, split, self.dataset_name)
            file_path = project_dir.joinpath(relative_path)
            self.dataset[split] = pd.read_pickle(file_path)
            self.full_dataset = self.full_dataset.append(self.dataset[split])

        self.logger.info('Loaded dataset {}!'.format(self.dataset_name))

    def initialize_encoder(self):
        """Initialize Encoder"""
        project_dir = Path(__file__).resolve().parents[2]
        relative_path = 'data/processed/{}/split/raw/train_data_{}.pkl' \
            .format(self.original_dataset_name, self.original_dataset_name)
        file_path = project_dir.joinpath(relative_path)
        df_orig_ds = pd.read_pickle(file_path)
        self.encoder = LabelEncoder()
        self.encoder.fit(df_orig_ds['category'].values)

        self.logger.info('Initialized encoder using {}!'.format(self.original_dataset_name))

    def load_configuration(self, path):
        """Load configuration defined in the json for which a path is provided"""
        with open(path) as json_file:
            configuration = json.load(json_file)
            self.logger.info('Loaded configuration from {}!'.format(path))

        self.name = configuration['name']
        self.type = configuration['type']
        self.dataset_name = configuration['dataset']
        self.original_dataset_name = configuration['original_dataset']
        self.model_path = configuration['model_path']
        self.model_name = configuration['model_name']

        if configuration['evalualte_on_full_dataset'] == 'True':
            self.evaluate_on_full_dataset = True
        else:
            self.evaluate_on_full_dataset = False


    def evaluate(self):

        result_collector = ResultCollector(self.dataset_name, self.type)

        eval = evaluation.TransformersEvaluator(self.dataset_name, self.name, self.encoder)
        trainer = Trainer(
            model=self.model,  # the instantiated 🤗 Transformers model to be trained
            compute_metrics=eval.compute_metrics_transformers
        )
        if self.evaluate_on_full_dataset:
            ds_eval = self.full_dataset
        else:
            ds_eval = self.dataset['test']

        texts = list(ds_eval['title'].values)
        labels = list(ds_eval['category'].values)
        le_dict = dict(zip(self.encoder.classes_, self.encoder.transform(self.encoder.classes_)))

        tokenizer = utils.provide_tokenizer(self.model_name)

        ds_wdc = CategoryDataset(texts, labels, tokenizer, le_dict)

        result_collector.results[self.name]= trainer.evaluate(ds_wdc)

        # Persist results
        timestamp = time.time()
        result_collector.persist_results(timestamp)
