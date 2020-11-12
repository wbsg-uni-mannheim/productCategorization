import logging
from pathlib import Path

import pandas as pd


class ExperimentRunner:

    def __init__(self, path, test, experiment_type):
        self.logger = logging.getLogger(__name__)
        self.test = test
        if test:
            self.logger.warning('Run in Testmode!')
        self.path = path

        self.experiment_type = experiment_type
        self.dataset = {}
        self.parameter = None
        self.dataset_name = None

    def __str__(self):
        output = 'Experiment runner for {} experiments on {} dataset with the following parameter: {}' \
            .format(self.experiment_type, self.dataset_name, self.parameter)
        return output

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
        """Run experiments - Implemented in child classes!"""
