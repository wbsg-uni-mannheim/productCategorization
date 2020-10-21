import unittest
from pathlib import Path
import os
from datetime import datetime
import time

from src.experiments.experiment_runner import ExperimentRunner
import pandas as pd


class TestMakeDataSet(unittest.TestCase):

    def setUp(self):
        # Set up - Retrieve all information from configuration file!
        project_dir = Path(__file__).resolve().parents[3]
        path_to_experiments = project_dir.joinpath('experiments/testing/icecat/configuration/dictionary_based.json')

        self.runner = ExperimentRunner(path_to_experiments)
        self.path_to_results = None

    def test_load_dict_experiments(self):
        """Test load dict experiments"""
        # Tests
        self.assertEqual('dict-based', self.runner.experiment_type)
        self.assertEqual('icecat', self.runner.dataset_name)
        self.assertIsNotNone(self.runner.parameter)

    def test_persist_results(self):
        """Test result persistence"""

        result = {'First Experiment': {'weighted_prec': 0.56,
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
        self.runner.persist_results(result, timestamp)
        # Tests
        project_dir = Path(__file__).resolve().parents[3]
        path_to_results = project_dir.joinpath('experiments/icecat/results/')
        self.assertEqual(True, os.path.exists(path_to_results))
        path_to_results = path_to_results.joinpath('{}_{}_results_{}.csv'.format(
            'icecat', 'dict-based', datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d_%H-%M-%S')))
        self.assertEqual(True, os.path.exists(path_to_results))

        df_results = pd.read_csv(path_to_results.absolute(), sep=';')

        self.assertEqual('First Experiment', df_results.loc[0]['Experiment Name'])
        self.assertEqual(0.58, df_results.loc[1]['weighted_prec'])
        self.assertEqual(0.74, df_results.loc[1]['h_f1'])
        self.assertEqual(6, len(df_results.columns))

        self.path_to_results = path_to_results

    def tearDown(self):
        # Delete generated files!
        if self.path_to_results:
            os.remove(self.path_to_results)
            self.path_to_results = None


if __name__ == '__main__':
    unittest.main()
