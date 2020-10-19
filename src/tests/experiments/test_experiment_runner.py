import unittest
from pathlib import Path

from src.experiments.experiment_runner import ExperimentRunner

class TestMakeDataSet(unittest.TestCase):

    def test_load_dict_experiments(self):
        """Test load dict experiments"""
        # Set up
        project_dir = Path(__file__).resolve().parents[3]
        path_to_experiments = project_dir.joinpath('experiments/icecat/dictionary_based.json')

        # Load experiments
        runner = ExperimentRunner(path_to_experiments)

        # Tests
        self.assertEqual('dict-based', runner.experiment_type)
        self.assertEqual('icecat', runner.dataset)
        self.assertIsNotNone(runner.parameter)


if __name__ == '__main__':
    unittest.main()