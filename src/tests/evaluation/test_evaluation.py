import unittest


from src.evaluation import evaluation


class TestMakeDataSet(unittest.TestCase):

    def test_compute_metrics_transformers(self):
        # Setup
        dataset_name = 'webdatacommons'
        experiment_name = 'test_compute_metrics'

        precision = 0.47
        recall = 0.4
        f1 = 0.43
        h_f1 = 0.5

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
        evaluator = evaluation.HierarchicalEvaluator(dataset_name, experiment_name, None)
        scores = evaluator.compute_metrics(labels, preds)

        self.assertEqual(precision, round(scores['weighted_prec'], 2))
        self.assertEqual(recall, round(scores['weighted_rec'], 2))
        self.assertEqual(f1, round(scores['weighted_f1'], 2))

        self.assertEqual(h_f1, round(scores['h_f1'], 2))



        #Evaluate Results -To-Do: Implement test and replace evaluation overhead afterwards.