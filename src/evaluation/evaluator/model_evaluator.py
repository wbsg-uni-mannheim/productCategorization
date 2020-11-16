from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.models.model_runner import ModelRunner


class ModelEvaluator(ModelRunner):

    def __init__(self, configuration_path, test, experiment_type):
        super().__init__(configuration_path, test, experiment_type)

        self.full_dataset = None
        self.model = None
        self.encoder = None

        self.original_dataset_name = None
        self.model_path = None
        self.model_name = None
        self.prediction_output = None
        self.evaluate_on_full_dataset = None
        self.experiment_name = None
        self.parameter = None

        self.load_experiments(configuration_path)
        self.load_datasets()

        # Create full (concatenated) dataset if necessary!
        if self.evaluate_on_full_dataset:
            self.create_full_dataset()

        self.initialize_encoder()  # Preprocess dataset labels as done for training dataset

    def create_full_dataset(self):
        """Load dataset for the given experiments"""
        self.full_dataset = pd.DataFrame()
        for split in self.dataset:
            self.full_dataset = self.full_dataset.append(self.dataset[split])

        self.logger.info('Created full dataset!')

    def initialize_encoder(self):
        """Initialize Encoder"""
        project_dir = Path(__file__).resolve().parents[3]
        relative_path = 'data/processed/{}/split/raw/train_data_{}.pkl' \
            .format(self.original_dataset_name, self.original_dataset_name)
        file_path = project_dir.joinpath(relative_path)
        df_orig_ds = pd.read_pickle(file_path)
        self.encoder = LabelEncoder()
        self.encoder.fit(df_orig_ds['category'].values)

        self.logger.info('Initialized encoder using {}!'.format(self.original_dataset_name))

    def load_experiments(self, relative_path):
        """Load configuration defined in the json for which a path is provided"""
        experiments = self.load_configuration(relative_path)

        self.experiment_name = experiments['experiment_name']
        self.original_dataset_name = experiments['original_dataset']
        self.model_path = experiments['model_path']
        self.model_name = experiments['model_name']
        self.prediction_output = experiments['prediction_output']

        if experiments['evaluate_on_full_dataset'] == 'True':
            self.evaluate_on_full_dataset = True
        else:
            self.evaluate_on_full_dataset = False

        self.parameter = experiments



    def evaluate(self):
        """Implemented in subclass"""

    def prepare_eval_dataset(self):
        if self.evaluate_on_full_dataset:
            ds_eval = self.full_dataset
        else:
            ds_eval = self.dataset['test']

        if self.test:
            # Load only a subset of the data
            ds_eval = ds_eval[:50]
            self.logger.warning('Run in test mode - dataset reduced to 50 records!')

        return ds_eval
