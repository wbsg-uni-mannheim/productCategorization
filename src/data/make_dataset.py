# -*- coding: utf-8 -*-
import os

import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import csv


@click.command()
@click.option('--dataset', help='Dataset which you like to preprocess')
def main(dataset):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    if dataset not in DATASETS:
        msg = 'Dataset {} not defined!'.format(dataset)
        logger.error(msg)
        raise ValueError(msg)

    else:
        load_and_split_dataset(dataset)


def load_and_split_dataset(dataset):
    logger = logging.getLogger(__name__)

    project_dir = Path(__file__).resolve().parents[2]

    path_to_raw_data = project_dir.joinpath('data/raw', dataset)

    if os.path.exists(path_to_raw_data.joinpath('split')):
        logger.info('Dataset {} is already split!'.format(dataset))

    else:
        logger.info('Dataset {} is not split yet!'.format(dataset))
        dataset_name = '{}_data_raw.csv'.format(dataset)
        path_to_raw_dataset = path_to_raw_data.joinpath(dataset_name)

        df_raw_data = pd.read_csv(path_to_raw_dataset.absolute(), sep=',', encoding='utf-8', quotechar='"', quoting=csv.QUOTE_ALL)
        logger.info('Dataset {} loaded!'.format(dataset))


def split_dataset(df_dataset):
    """Split dataset into train, test and validation set"""
# split into training and testing set 80:20 (stratify) and then again 80:20 into training and validation
# which makes it 64% training, 16% validation, 20% testing
#icecat_data_train, icecat_data_test, icecat_data_train_target, icecat_data_test_target = train_test_split(
#    icecat_data_full, icecat_data_full[['pathlist_names', 'Category.Name.Value', 'Category.CategoryID']], test_size=0.2,
#    random_state=42,
#    shuffle=True, stratify=icecat_data_full[['pathlist_names']])

#icecat_data_train, icecat_data_validate, icecat_data_train_target, icecat_data_validate_target = train_test_split(
#    icecat_data_train, icecat_data_train_target, test_size=0.2, random_state=42)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    DATASETS = os.getenv("DATASETS")

    main()
