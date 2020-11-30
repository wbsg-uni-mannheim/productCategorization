import unittest
from pathlib import Path

from src.experiments.runner.experiment_runner_dict import ExperimentRunnerDict


class TestExperimentRunner(unittest.TestCase):

    def setUp(self):
        # Set up - Retrieve all information from configuration file!
        path_to_experiments = 'experiments/testing/icecat/dictionary_based_models.json'

        self.runner = ExperimentRunnerDict(path_to_experiments, True, 'dict-based')
        self.path_to_results = None

    def test_load_dict_experiments(self):
        """Test load dict experiments"""
        # Tests
        self.assertEqual('dict-based', self.runner.experiment_type)
        self.assertEqual('icecat', self.runner.dataset_name)
        self.assertIsNotNone(self.runner.parameter)


if __name__ == '__main__':
    unittest.main()
