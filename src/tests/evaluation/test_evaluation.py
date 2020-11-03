import unittest

from src.evaluation import evaluation


class TestMakeDataSet(unittest.TestCase):

    def test_compute_metrics_transformers(self):
        # Setup
        dataset_name = 'webdatacommons'
        experiment_name = 'test_compute_metrics_transformers'

        #Run Function
        eval = evaluation.TransformersEvaluator(dataset_name, experiment_name, None)
        #eval.compute_metrics_transformers()

        #Evaluate Results -To-Do: Implement test and replace evaluation overhead afterwards.