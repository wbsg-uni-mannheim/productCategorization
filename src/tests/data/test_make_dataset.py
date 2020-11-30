from src.data.make_dataset import trigger_load_dataset

import os
import unittest
import pandas as pd
from pathlib import Path


class TestMakeDataSet(unittest.TestCase):

    def test_unknown_dataset(self):
        """Try to load unknown data set"""
        self.assertRaises(ValueError, trigger_load_dataset, 'unknown')

    def test_load_dataset(self):
        """Test data load with existing data set --> Subset of Rakuten data set"""
        # Set up
        path_to_pickle = 'data/processed/subset_rakuten/split/raw/train_data_subset_rakuten.pkl'

        # Execute function
        trigger_load_dataset('subset_rakuten')
        # Test success
        self.assertEqual(True, os.path.exists(path_to_pickle))

        df_train_rakuten = pd.read_pickle(path_to_pickle)
        df_testing_train_rakuten = pd.read_pickle('data/testing/processed/subset_rakuten/split/raw/subset_rakuten_data_train.pkl')

        # Compare number of columns
        self.assertEqual(len(df_testing_train_rakuten.columns), len(df_train_rakuten.columns))

        # Compare naming of columns
        for value in df_train_rakuten.columns:
            self.assertIn(value, df_testing_train_rakuten.columns)

        # Compare values
        for index, row in df_train_rakuten.iterrows():
            self.assertEqual(df_testing_train_rakuten.loc[index]['title'], row['title'])
            self.assertEqual(str(df_testing_train_rakuten.loc[index]['category']), str(row['category']))
            self.assertEqual(str(df_testing_train_rakuten.loc[index]['path_list']), str(row['path_list']))


if __name__ == '__main__':
    unittest.main()
