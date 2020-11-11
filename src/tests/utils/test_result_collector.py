import os
import time

import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.utils.result_collector import ResultCollector


class TestResultCollector(unittest.TestCase):

    def setUp(self):
        self.path_to_results = None

    def tearDown(self):
        # Delete generated files!
        if self.path_to_results:
            os.remove(self.path_to_results)
            self.path_to_results = None

    def test_persist_results(self):
        """Test result persistence"""
        # Setup
        dataset_name = 'icecat'
        experiment_type = 'dict_based'
        result_collector = ResultCollector(dataset_name, experiment_type)

        result_collector.results = {'First Experiment+train': {'weighted_prec': 0.56,
                                       'weighted_rec': 0.56,
                                       'weighted_f1': 0.56,
                                       'macro_f1': 0.54,
                                       'h_f1': 0.34},
                  'Second Experiment': {'weighted_prec': 0.58,
                                        'weighted_rec': 0.83,
                                        'weighted_f1': 0.76,
                                        'macro_f1': 0.64,
                                        'h_f1': 0.74}}

        timestamp = time.time()
        result_collector.persist_results(timestamp)
        # Tests
        project_dir = Path(__file__).resolve().parents[3]
        path_to_results = project_dir.joinpath('experiments/{}/results/'.format(dataset_name))
        self.assertEqual(True, os.path.exists(path_to_results))
        path_to_results = path_to_results.joinpath('{}_{}_results_{}.csv'.format(
            dataset_name, experiment_type, datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d_%H-%M-%S')))
        self.assertEqual(True, os.path.exists(path_to_results))

        df_results = pd.read_csv(path_to_results.absolute(), sep=';')

        self.assertEqual('First Experiment', df_results.loc[0]['Experiment Name'])
        self.assertEqual('train', df_results.loc[0]['Split'])
        self.assertEqual(0.58, df_results.loc[1]['weighted_prec'])
        self.assertEqual(0.74, df_results.loc[1]['h_f1'])
        self.assertEqual(8, len(df_results.columns))

        self.path_to_results = path_to_results