#!/usr/bin/env python3

import json
import logging
from datetime import datetime
from pathlib import Path
import os
import pandas as pd
import csv
import time

import click


def load_results(datasets):
    """Load results for the given data set"""
    logger = logging.getLogger(__name__)

    project_dir = Path(__file__).resolve().parents[2]

    aggregated_results = pd.DataFrame()
    for dataset_name in datasets:
        relative_path = 'results/{}/'\
            .format(dataset_name)
        file_path = project_dir.joinpath(relative_path)

        for filename in os.listdir(file_path):
            if 'aggregated_results' not in filename:
                full_file_path = file_path.joinpath(filename)
                df_result = pd.read_csv(full_file_path, sep=';')
                df_result['filename'] = filename
                logger.info('Loaded results of experiment {}!'.format(filename))

                aggregated_results = aggregated_results.append(df_result, ignore_index=True)

    return aggregated_results

def append_original_configuration(df_results):
    """Append the configuration of the original data set"""
    df_results_with_configuration = pd.DataFrame()

    for index, row in df_results.iterrows():
        configuration = load_configuration(row['Dataset'], row['Experiment Name'])
        if 'eval' in configuration['type']:
            #Load configuration of original experiment if present
            configuration = \
                load_configuration(configuration['original_dataset'], configuration['original_experiment_name'])

        parameters = configuration["parameter"]
        for parameter in parameters:
            if parameter == 'experiment_name':
                row['Original Experiment Name'] = parameters[parameter]
            else:
                row[parameter] = parameters[parameter]

        df_results_with_configuration = df_results_with_configuration.append(row)

    return df_results_with_configuration

def load_configuration(dataset, experiment_name):
    logger = logging.getLogger(__name__)
    project_dir = Path(__file__).resolve().parents[2]

    relative_path = 'experiments/{}/{}.json'.format(dataset, experiment_name)
    absolute_path = project_dir.joinpath(relative_path)
    with open(absolute_path) as json_file:
        try:
            experiments = json.load(json_file)
        except FileNotFoundError as err:
            logger.error(err)

    return experiments

def persist_aggregated_results(df_results, datasets):
    """Persist aggregated results"""
    logger = logging.getLogger(__name__)

    project_dir = Path(__file__).resolve().parents[2]
    timestamp = time.time()
    string_timestap = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d_%H-%M-%S')
    datasets_string = '-'.join(datasets)
    relative_path = 'results/general/aggregated_results-{}-{}.csv' \
        .format(datasets_string, string_timestap)
    file_path = project_dir.joinpath(relative_path)

    df_results.to_csv(file_path, sep=';', encoding='utf-8', quotechar='"', quoting=csv.QUOTE_ALL, index=False)
    logger.info('Results aggregated into {}!'.format(relative_path))


@click.command()
@click.option('--dataset', multiple=True, help='Data sets for which the results will be aggregated!')
def main(dataset):
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    df_results = load_results(dataset)

    df_results_with_configuration = append_original_configuration(df_results)

    persist_aggregated_results(df_results_with_configuration, dataset)



if __name__ == '__main__':
    main()