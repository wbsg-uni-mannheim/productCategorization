import logging
import pickle
import unittest
from pathlib import Path

from src.evaluation import scorer


class TestMakeDataSet(unittest.TestCase):

    def test_compute_metrics(self):
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

        self.assertEqual(precision, round(scores['leaf_weighted_prec'], 2))
        self.assertEqual(recall, round(scores['leaf_weighted_rec'], 2))
        self.assertEqual(f1, round(scores['leaf_weighted_f1'], 2))

        self.assertEqual(h_f1, round(scores['h_f1'], 2))

    def test_compute_metrics_no_encoding(self):

        dataset_name = 'wdc_ziqi'
        experiment_name = 'test_compute_metrics_no_encoding'

        project_dir = Path(__file__).resolve().parents[3]
        path_to_tree = project_dir.joinpath('data', 'raw', dataset_name, 'tree', 'tree_{}.pkl'.format(dataset_name))

        with open(path_to_tree, 'rb') as f:
            tree = pickle.load(f)

        y_true = ['51121600_Vitamins/Minerals/Nutritional_Supplements', '64010200_Personal_Carriers/Accessories',
         '67010800_Upper_Body_Wear/Tops', '67010800_Upper_Body_Wear/Tops', '86011100_Toy_Vehicles_�_Non-ride',
         '67010800_Upper_Body_Wear/Tops', '75010300_Household/Office_Tables/Desks', '75030100_Ornaments',
         '64010200_Personal_Carriers/Accessories',
         '79010700_Plumbing/Heating_Ventilation/Air_Conditioning_Variety_Packs', '67010300_Lower_Body_Wear/Bottoms',
         '67010300_Lower_Body_Wear/Bottoms', '67010800_Upper_Body_Wear/Tops',
         '50202300_Non_Alcoholic_Beverages_�_Ready_to_Drink', '67010800_Upper_Body_Wear/Tops',
         '10101600_Pet_Nutritional_Supplements', '77030100_Cars', '67010100_Clothing_Accessories',
         '65010100_Computer/Video_Game_Accessories', '67010800_Upper_Body_Wear/Tops']

        y_pred = ['51121600_Vitamins/Minerals/Nutritional_Supplements', '51101600_Drug_Administration',
         '67010800_Upper_Body_Wear/Tops', '67010100_Clothing_Accessories', '70011300_Arts/Crafts_Variety_Packs',
         '67010800_Upper_Body_Wear/Tops', '75010300_Household/Office_Tables/Desks', '75030200_Pictures/Mirrors/Frames',
         '64010200_Personal_Carriers/Accessories',
         '79010700_Plumbing/Heating_Ventilation/Air_Conditioning_Variety_Packs', '67010300_Lower_Body_Wear/Bottoms',
         '67010300_Lower_Body_Wear/Bottoms', '67010200_Full_Body_Wear', '50193800_Ready-Made_Combination_Meals',
         '67010800_Upper_Body_Wear/Tops', '86010400_Developmental/Educational_Toys', '77030100_Cars',
         '67010100_Clothing_Accessories', '71011600_Sporting_Firearms_Equipment', '67010800_Upper_Body_Wear/Tops']

        evaluator = scorer.HierarchicalScorer(experiment_name, tree)
        results = evaluator.compute_metrics_no_encoding(y_true, y_pred)

        # Lvl3 must match leaf node prediction in this scenario
        self.assertEqual(results['weighted_f1_lvl_3'], results['leaf_weighted_f1'])


