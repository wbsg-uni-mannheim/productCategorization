import logging
import os
from pathlib import Path

import click
import pandas as pd

from src.data.preprocessing import preprocess

@click.command()
@click.option('--dataset_name', help='Dataset which you like to prepare for language modelling')
@click.option('--additional_ds_path', help='Additional dataset for language modelling')
@click.option('--additional_ds_suffix', help='Suffix to identify the additional ds')
def main(dataset_name, additional_ds_path, additional_ds_suffix):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    dataset = load_dataset(dataset_name)

    #Check if additional dataset information is provided
    if not (additional_ds_path is None) and not (additional_ds_suffix is None):
        df_additional_ds = pd.read_csv(additional_ds_path, sep=';')
    else:
        df_additional_ds = None
    generate_datasets_for_language_modelling(dataset, dataset_name, df_additional_ds, additional_ds_suffix)


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


def generate_datasets_for_language_modelling(dataset, dataset_name, df_additional_ds, additional_ds_suffix):
    logger = logging.getLogger(__name__)
    data_dir = os.environ['DATA_DIR']
    data_dir = Path(data_dir)

    configurations = []
    config_1 = {'category': True, 'category_reverse': False, 'description': True,
                'multiple_rows': True, 'additional_ds': True}
    configurations.append(config_1)

    for config in configurations:
        # Make sure that an additional dataset is properly provided if requested
        if df_additional_ds is None and additional_ds_suffix:
            config['additional_ds'] = False
        generate_and_store_single_dataset_for_language_modelling(dataset, dataset_name, data_dir, config, df_additional_ds, additional_ds_suffix)

def generate_and_store_single_dataset_for_language_modelling(dataset, dataset_name, data_dir, config, df_additional_ds, additional_ds_suffix):
    logger = logging.getLogger(__name__)

    suffix = 'title'
    for key in config:
        if config[key]:
            suffix = '{}_{}'.format(suffix, key)

    if not (additional_ds_suffix is None) and config['additional_ds']:
        suffix = '{}_{}'.format(suffix, additional_ds_suffix)

    for split in dataset:
        relative_path = 'data/processed/{}/language-modelling/{}_language_modelling_{}_with_{}.txt'.format(dataset_name, split, dataset_name, suffix)
        file_path = data_dir.joinpath(relative_path)
        with open(file_path, 'w') as file:
            for index, row in dataset[split].iterrows():
                #preprocess values
                prep_title = preprocess(row['title'])
                line = '{}'.format(prep_title)

                if config['category']:
                    categories = row['path_list'].split('>')
                    categories = [value.split('_')[1] for value in categories]
                    categories = [preprocess(value) for value in categories]

                    new_line = prepare_category(config,categories,line)

                    if config['multiple_rows']:
                        write_dataset_to_file(file,new_line)
                    else:
                        line = new_line

                if config['description']:
                    new_line = prepare_description(row['description'], line)

                    if config['multiple_rows']:
                        write_dataset_to_file(file, new_line)
                    else:
                        line = new_line

                if not config['multiple_rows']:
                    write_dataset_to_file(file, line)

                if split == 'train' and config['additional_ds']:
                    for index, row in df_additional_ds.iterrows():
                        line = preprocess(row['Title'])
                        categories = []
                        if row['Category'] is not None and type(row['Category']) is str:
                            categories.append(row['Category'])
                        if row['Breadcrumb'] is not None and type(row['Breadcrumb']) is str:
                            categories.append(row['Breadcrumb'])
                        if row['BreadcrumbList'] is not None and type(row['BreadcrumbList']) is str:
                            categories.append(row['BreadcrumbList'])


                        if len(categories) > 0 and config['category']:
                            new_line = prepare_category(config, categories, line)

                            if config['multiple_rows']:
                                write_dataset_to_file(file, new_line)
                            else:
                                line = new_line

                        if type(row['Description']) is str and config['description']:
                            new_line = prepare_description(row['Description'], line)

                            if config['multiple_rows']:
                                write_dataset_to_file(file, new_line)
                            else:
                                line = new_line
                        file.write('{}\n'.format(line))

        logger.info('File {} created for Language Modelling!'.format(relative_path))

def prepare_category(config, categories, line):
    if config['category_reverse']:
        categories.reverse()

    prep_catgories = ' '.join(categories)
    new_line = '{} - {}'.format(line, prep_catgories)

    return new_line

def prepare_description(description, line):
    description_values = description.split('.')
    preprocessed_description_values = []
    for value in description_values:
        if len(value) > 4:
            preprocessed_description_values.append(preprocess(value))

    new_line = '{} - {}'.format(line, '. '.join(preprocessed_description_values))

    return new_line

def write_dataset_to_file(file, line):
    line = '{}\n'.format(line)
    file.write(line)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()