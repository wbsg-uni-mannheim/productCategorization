import logging
import os
from pathlib import Path

import click
import pandas as pd

from src.data.preprocessing import preprocess


@click.command()
@click.option('--dataset_name', help='Dataset which you like to prepare for language modelling')
def main(dataset_name):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    dataset = load_dataset(dataset_name)
    store_dataset_for_language_modelling(dataset, dataset_name)


def load_dataset(dataset_name):
    """Load dataset for the given experiments"""
    logger = logging.getLogger(__name__)
    data_dir = os.environ['DATA_DIR']
    data_dir = Path(data_dir)
    splits = ['train', 'validate']
    dataset = {}

    for split in splits:
        relative_path = 'data/processed/{}/split/raw/{}_data_{}.pkl'.format(dataset_name, split, dataset_name)
        file_path = data_dir.joinpath(relative_path)
        dataset[split] = pd.read_pickle(file_path)

    logger.info('Loaded dataset {}!'.format(dataset_name))

    return dataset


def store_dataset_for_language_modelling(dataset, dataset_name):
    logger = logging.getLogger(__name__)
    data_dir = os.environ['DATA_DIR']
    data_dir = Path(data_dir)

    for split in dataset:
        relative_path = 'data/processed/{}/language-modelling/{}_language_modelling_{}.txt'.format(dataset_name, split, dataset_name)
        file_path = data_dir.joinpath(relative_path)
        with open(file_path, 'w') as file:
            for title in dataset[split]['title'].values:
                #preprocess value
                prep_title = preprocess(title)
                file.write(prep_title + '\n')

    logger.info('Dataset prepared for language modelling {}!'.format(dataset_name))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()