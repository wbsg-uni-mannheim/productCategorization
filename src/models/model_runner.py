import json
import logging
from pathlib import Path

import pandas as pd


class ModelRunner:

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

    def load_configuration(self, path):
        with open(path) as json_file:
            experiments = json.load(json_file)
            self.logger.info('Loaded experiments from {}!'.format(path))

        if self.experiment_type != experiments['type']:
            raise ValueError('Run experiment type and experiment type from {} do not match!'.format(path))
        self.dataset_name = experiments['dataset']

        return experiments

    def load_datasets(self):
        """Load dataset for the given experiments"""
        project_dir = Path(__file__).resolve().parents[2]
        splits = ['train', 'validate', 'test']

        for split in splits:
            relative_path = 'data/processed/{}/split/raw/{}_data_{}.pkl'.format(self.dataset_name, split, self.dataset_name)
            file_path = project_dir.joinpath(relative_path)
            self.dataset[split] = pd.read_pickle(file_path)

        self.logger.info('Loaded dataset {}!'.format(self.dataset_name))
