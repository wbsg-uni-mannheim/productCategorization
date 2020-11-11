import unittest
from pathlib import Path

from src.experiments.experiment_runner import ExperimentRunner



class TestExperimentRunner(unittest.TestCase):

    def setUp(self):
        # Set up - Retrieve all information from configuration file!
        project_dir = Path(__file__).resolve().parents[3]
        path_to_experiments = project_dir.joinpath('experiments/testing/icecat/configuration/dictionary_based_models.json')

        self.runner = ExperimentRunner(path_to_experiments, True)
        self.path_to_results = None

    def test_load_dict_experiments(self):
        """Test load dict experiments"""
        # Tests
        self.assertEqual('dict-based', self.runner.experiment_type)
        self.assertEqual('icecat', self.runner.dataset_name)
        self.assertIsNotNone(self.runner.parameter)


if __name__ == '__main__':
    unittest.main()
