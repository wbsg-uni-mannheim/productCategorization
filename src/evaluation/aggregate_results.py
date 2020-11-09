#!/usr/bin/env python3

import logging
from datetime import datetime
from pathlib import Path
import os
import pandas as pd
import csv
import time

import click


def load_results(dataset_name):
    """Load results for the given data set"""
    logger = logging.getLogger(__name__)

    project_dir = Path(__file__).resolve().parents[2]

    relative_path = 'experiments/{}/results/'\
        .format(dataset_name)
    file_path = project_dir.joinpath(relative_path)

    aggregated_results = pd.DataFrame()
    for filename in os.listdir(file_path):
        if 'aggregated_results' not in filename:
            full_file_path = file_path.joinpath(filename)
            df_result = pd.read_csv(full_file_path, sep=';')
            df_result['filename'] = filename
            logger.info('Loaded results of experiment {}!'.format(filename))
            ds_name = experiments['dataset']


            experiment_type = experiments['type']

            aggregated_results = aggregated_results.append(df_result, ignore_index=True)

    return aggregated_results

def persist_aggregated_results(df_results, dataset_name):
    """Persist aggregated results"""
    logger = logging.getLogger(__name__)

    project_dir = Path(__file__).resolve().parents[2]
    timestamp = time.time()
    string_timestap = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d_%H-%M-%S')
    relative_path = 'experiments/{}/results/aggregated_results-{}-{}.csv' \
        .format(dataset_name, dataset_name, string_timestap)
    file_path = project_dir.joinpath(relative_path)

    df_results.to_csv(file_path, sep=';', encoding='utf-8', quotechar='"', quoting=csv.QUOTE_ALL)
    logger.info('Results aggregated into {}!'.format(relative_path))


@click.command()
@click.option('--dataset_name', help='Data Set to aggregate results')
def main(dataset_name):
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    df_results = load_results(dataset_name)

    #To-Do: Enrich with config file!

    persist_aggregated_results(df_results, dataset_name)



if __name__ == '__main__':
    main()