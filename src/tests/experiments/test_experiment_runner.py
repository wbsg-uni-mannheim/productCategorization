import unittest
from pathlib import Path

from src.experiments.experiment_runner_dict import ExperimentRunnerDict


class TestExperimentRunner(unittest.TestCase):

    def setUp(self):
        # Set up - Retrieve all information from configuration file!
        project_dir = Path(__file__).resolve().parents[3]
        path_to_experiments = project_dir.joinpath('experiments/testing/icecat/configuration/dictionary_based_models.json')

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
