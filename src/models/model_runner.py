import json
import logging
import pickle
from pathlib import Path
from sys import platform

import os
import pandas as pd

from src.utils.tree_utils import TreeUtils


class ModelRunner:

    def __init__(self, path, test, experiment_type):

        #Initialize logging
        self.initialize_logging(path)
        self.logger = logging.getLogger(__name__)

        self.test = test
        if test:
            self.logger.warning('Run in Testmode!')
        self.path = path
        self.data_dir = os.environ['DATA_DIR']

        self.experiment_type = experiment_type
        self.dataset = {}
        self.parameter = None
        self.dataset_name = None

        self.tree = None
        self.root = None

    def initialize_logging(self, path):
        # Extract experiment name from config for logging
        if platform == "win32" and '\\' in path:
            config_path = path.split('\\')
        else:
            config_path = path.split('/')
        dataset = config_path[-2]
        experiment_name = config_path[-1].split('.')[0]

        log_file = '{}_{}.log'.format(dataset, experiment_name)
        log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO, format=log_fmt)

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
        splits = ['train', 'validate', 'test']

        for split in splits:
            relative_path = 'data/processed/{}/split/raw/{}_data_{}.pkl'.format(self.dataset_name, split, self.dataset_name)
            data_dir = Path(self.data_dir)
            file_path = data_dir.joinpath(relative_path)
            self.dataset[split] = pd.read_pickle(file_path)

        self.logger.info('Loaded dataset {}!'.format(self.dataset_name))

    def load_tree(self):
        data_dir = Path(self.data_dir)
        path_to_tree = data_dir.joinpath('data', 'raw', self.dataset_name, 'tree', 'tree_{}.pkl'.format(self.dataset_name))

        with open(path_to_tree, 'rb') as f:
            self.tree = pickle.load(f)
            self.logger.info('Loaded tree for dataset {}!'.format(self.dataset_name))

        self.root = [node[0] for node in self.tree.in_degree if node[1] == 0][0]

        self.tree_utils = TreeUtils(self.tree)

