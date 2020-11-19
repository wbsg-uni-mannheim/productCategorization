import pickle
import unittest
from pathlib import Path

from src.evaluation import scorer


class TestMakeDataSet(unittest.TestCase):

    def test_compute_metrics_transformers(self):
        # Setup
        dataset_name = 'icecat'
        experiment_name = 'test_compute_metrics'

        project_dir = Path(__file__).resolve().parents[3]
        path_to_tree = project_dir.joinpath('data', 'raw', dataset_name, 'tree', 'tree_{}.pkl'.format(dataset_name))

        with open(path_to_tree, 'rb') as f:
            tree = pickle.load(f)

        precision = 0.47
        recall = 0.4
        f1 = 0.43
        h_f1 = 0.45

        labels = ['Notebooks', 'Ink Cartridges', 'Toner Cartridges', 'Notebooks', 'Notebooks', 'Servers',
                  'Motherboards', 'Notebook Spare Parts', 'Warranty & Support Extensions', 'Fibre Optic Cables',
                  'Notebook Spare Parts', 'Toner Cartridges', 'Digital Photo Frames', 'Notebooks',
                  'Notebook Spare Parts', 'Notebooks', 'Notebooks', 'PCs/Workstations', 'PCs/Workstations',
                  'Notebook Cases']
        preds = ['IT Courses', 'Notebooks', 'Toner Cartridges', 'AV Extenders', 'Notebooks', 'Servers',
                 'Other Input Devices', 'Notebooks', 'Warranty & Support Extensions', 'Fibre Optic Cables',
                 'Cable Splitters or Combiners', 'Stick PCs', 'Digital Photo Frames', 'Notebooks', 'Notebooks',
                 'Projection Screens', 'Cable Splitters or Combiners', 'Cable Splitters or Combiners',
                 'Cable Splitters or Combiners', 'Notebook Cases']

        #Run Function
        evaluator = scorer.HierarchicalScorer(experiment_name, tree)

        decoder = dict(tree.nodes(data="name"))
        encoder = dict([(value, key) for key, value in decoder.items()])

        labels = [encoder[label] for label in labels]
        preds = [encoder[pred] for pred in preds]

        # To-Do: change input [labels], [preds](!)
        scores = evaluator.compute_metrics(labels, preds)

        self.assertEqual(precision, round(scores['weighted_prec'], 2))
        self.assertEqual(recall, round(scores['weighted_rec'], 2))
        self.assertEqual(f1, round(scores['weighted_f1'], 2))

        self.assertEqual(h_f1, round(scores['h_f1'], 2))
