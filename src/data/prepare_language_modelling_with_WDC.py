import logging
import os
from pathlib import Path

import click
import pandas as pd

from src.data.preprocessing import preprocess


@click.command()
@click.option('--dataset_name', help='Dataset which you like to prepare for language modelling')
@click.option('--file_path', help='Additional dataset for language modelling')
@click.option('--suffix', help='Suffix for generated datasets')
def main(dataset_name, file_path, suffix):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    dataset = load_dataset(dataset_name)
    df_additional = pd.read_csv(file_path, sep=';')
    store_dataset_for_language_modelling(dataset, dataset_name, df_additional, suffix)


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

def store_dataset_for_language_modelling(dataset, dataset_name, df_additional, suffix):
    logger = logging.getLogger(__name__)
    data_dir = os.environ['DATA_DIR']
    data_dir = Path(data_dir)

    for split in dataset:
    #     relative_path = 'data/processed/{}/language-modelling/{}_language_modelling_{}_additional_ds_{}.txt'.format(dataset_name, split, dataset_name, suffix)
    #     file_path = data_dir.joinpath(relative_path)
    #     with open(file_path, 'w') as file:
    #         for title in dataset[split]['title'].values:
    #             #preprocess value
    #             prep_title = preprocess(title)
    #             line = '{} \n'.format(prep_title)
    #             file.write(line)
    #
    #         if split == 'train':
    #             for title in df_additional['Title'].values:
    #                 line = '{} \n'.format(title)
    #                 file.write(line)
    #
    #     logger.info('File {} created for Language Modelling!'.format(relative_path))
    #
    #     # relative_path = 'data/processed/{}/language-modelling/{}_language_modelling_{}_with_category.txt'.format(dataset_name, split, dataset_name)
    #     # file_path = data_dir.joinpath(relative_path)
    #     # with open(file_path, 'w') as file:
    #     #     for index, row in dataset[split].iterrows():
    #     #         #preprocess value
    #     #         prep_title = preprocess(row['title'])
    #     #         categories = row['path_list'].split('>')
    #     #         categories = [value.split('_')[1] for value in categories]
    #     #         categories = [preprocess(value) for value in categories]
    #     #         prep_catgories = ' '.join(categories)
    #     #         line = '{} {}\n'.format(prep_title, prep_catgories)
    #     #         file.write(line)
    #
    #     relative_path = 'data/processed/{}/language-modelling/{}_language_modelling_{}_with_category_as_sentence_additional_ds_{}.txt'.format(dataset_name, split, dataset_name, suffix)
    #     file_path = data_dir.joinpath(relative_path)
    #     with open(file_path, 'w') as file:
    #
    #         for index, row in dataset[split].iterrows():
    #             #preprocess value
    #             prep_title = preprocess(row['title'])
    #             categories = row['path_list'].split('>')
    #             categories = [value.split('_')[1] for value in categories]
    #             categories = [preprocess(value) for value in categories]
    #             prep_catgories = ', '.join(categories)
    #             line = '{} belongs to {}.\n'.format(prep_title, prep_catgories)
    #             file.write(line)
    #
    #         if split == 'train':
    #             for index, row in df_additional.iterrows():
    #                 categories = []
    #                 if row['Category'] is not None and type(row['Category']) is str:
    #                     categories.append(row['Category'])
    #                 if row['Breadcrumb'] is not None and type(row['Breadcrumb']) is str:
    #                     categories.append(row['Breadcrumb'])
    #                 if row['BreadcrumbList'] is not None and type(row['BreadcrumbList']) is str:
    #                     categories.append(row['BreadcrumbList'])
    #
    #                 if len(categories) > 0:
    #                     prep_categories = ' '.join(categories).strip()
    #
    #                     line = '{} belongs to {}.\n'.format(row['Title'], prep_categories)
    #                     file.write(line)
    #
    #     logger.info('File {} created for Language Modelling!'.format(relative_path))
    #
    #     relative_path = 'data/processed/{}/language-modelling/{}_language_modelling_{}_with_category_reverse_as_sentence_additional_ds_{}.txt'.format(dataset_name, split, dataset_name, suffix)
    #     file_path = data_dir.joinpath(relative_path)
    #     with open(file_path, 'w') as file:
    #
    #         for index, row in dataset[split].iterrows():
    #             #preprocess value
    #             prep_title = preprocess(row['title'])
    #             categories = row['path_list'].split('>')
    #             categories = [value.split('_')[1] for value in categories]
    #             categories = [preprocess(value) for value in categories]
    #             categories.reverse()
    #             prep_catgories = ', '.join(categories)
    #             line = '{} belongs to {}.\n'.format(prep_title, prep_catgories)
    #             file.write(line)
    #
    #         if split == 'train':
    #             for index, row in df_additional.iterrows():
    #                 categories = []
    #                 if row['Category'] is not None and type(row['Category']) is str:
    #                     categories.append(row['Category'])
    #                 if row['Breadcrumb'] is not None and type(row['Breadcrumb']) is str:
    #                     categories.append(row['Breadcrumb'])
    #                 if row['BreadcrumbList'] is not None and type(row['BreadcrumbList']) is str:
    #                     categories.append(row['BreadcrumbList'])
    #
    #                 if len(categories) > 0:
    #                     categories.reverse()
    #                     prep_categories = ' '.join(categories).strip()
    #
    #                     line = '{} belongs to {}.\n'.format(row['Title'], prep_categories)
    #                     file.write(line)
    #
    #     logger.info('File {} created for Language Modelling!'.format(relative_path))
    #
    #     relative_path = 'data/processed/{}/language-modelling/{}_language_modelling_{}_with_category_additional_ds_{}.txt'.format(dataset_name, split, dataset_name, suffix)
    #     file_path = data_dir.joinpath(relative_path)
    #     with open(file_path, 'w') as file:
    #
    #         for index, row in dataset[split].iterrows():
    #             #preprocess value
    #             prep_title = preprocess(row['title'])
    #             categories = row['path_list'].split('>')
    #             categories = [value.split('_')[1] for value in categories]
    #             categories = [preprocess(value) for value in categories]
    #             prep_catgories = ', '.join(categories)
    #             line = '{} - {}.\n'.format(prep_title, prep_catgories)
    #             file.write(line)
    #
    #         if split == 'train':
    #             for index, row in df_additional.iterrows():
    #                 categories = []
    #                 if row['Category'] is not None and type(row['Category']) is str:
    #                     categories.append(row['Category'])
    #                 if row['Breadcrumb'] is not None and type(row['Breadcrumb']) is str:
    #                     categories.append(row['Breadcrumb'])
    #                 if row['BreadcrumbList'] is not None and type(row['BreadcrumbList']) is str:
    #                     categories.append(row['BreadcrumbList'])
    #
    #                 if len(categories) > 0:
    #                     prep_categories = ' '.join(categories).strip()
    #
    #                     line = '{} - {}.\n'.format(row['Title'], prep_categories)
    #                     file.write(line)
    #
    #     logger.info('File {} created for Language Modelling!'.format(relative_path))
    #
    #     relative_path = 'data/processed/{}/language-modelling/{}_language_modelling_{}_with_category_reverse_additional_ds_{}.txt'.format(dataset_name, split, dataset_name, suffix)
    #     file_path = data_dir.joinpath(relative_path)
    #     with open(file_path, 'w') as file:
    #
    #         for index, row in dataset[split].iterrows():
    #             #preprocess value
    #             prep_title = preprocess(row['title'])
    #             categories = row['path_list'].split('>')
    #             categories = [value.split('_')[1] for value in categories]
    #             categories = [preprocess(value) for value in categories]
    #             categories.reverse()
    #             prep_catgories = ', '.join(categories)
    #             line = '{} - {}.\n'.format(prep_title, prep_catgories)
    #             file.write(line)
    #
    #         if split == 'train':
    #             for index, row in df_additional.iterrows():
    #                 categories = []
    #                 if row['Category'] is not None and type(row['Category']) is str:
    #                     categories.append(row['Category'])
    #                 if row['Breadcrumb'] is not None and type(row['Breadcrumb']) is str:
    #                     categories.append(row['Breadcrumb'])
    #                 if row['BreadcrumbList'] is not None and type(row['BreadcrumbList']) is str:
    #                     categories.append(row['BreadcrumbList'])
    #
    #                 if len(categories) > 0:
    #                     categories.reverse()
    #                     prep_categories = ' '.join(categories).strip()
    #
    #                     line = '{} - {}.\n'.format(row['Title'], prep_categories)
    #                     file.write(line)
    #
    #     logger.info('File {} created for Language Modelling!'.format(relative_path))
    #
    #     # relative_path = 'data/processed/{}/language-modelling/{}_language_modelling_{}_with_only_one_category_as_sentence.txt'.format(dataset_name, split, dataset_name)
    #     # file_path = data_dir.joinpath(relative_path)
    #     # with open(file_path, 'w') as file:
    #     #
    #     #     for index, row in dataset[split].iterrows():
    #     #         #preprocess value
    #     #         prep_title = preprocess(row['title'])
    #     #         categories = row['path_list'].split('>')
    #     #         categories = [value.split('_')[1] for value in categories]
    #     #         categories = [preprocess(value) for value in categories]
    #     #         for category in categories:
    #     #             line = '{} belongs to {}.\n'.format(prep_title, category)
    #     #             file.write(line)
    #
    #     # relative_path = 'data/processed/{}/language-modelling/{}_language_modelling_{}_with_category_as_sentence_reverse_oder.txt'.format(dataset_name, split, dataset_name)
    #     # file_path = data_dir.joinpath(relative_path)
    #     # with open(file_path, 'w') as file:
    #     #
    #     #     for index, row in dataset[split].iterrows():
    #     #         #preprocess value
    #     #         prep_title = preprocess(row['title'])
    #     #         categories = row['path_list'].split('>')
    #     #         categories = [value.split('_')[1] for value in categories]
    #     #         categories = [preprocess(value) for value in categories]
    #     #         prep_catgories = ', '.join(categories)
    #     #         line = '{} belongs to {}.\n'.format(prep_title, prep_catgories)
    #     #         file.write(line)
    #
    # #     relative_path = 'data/processed/{}/language-modelling/{}_language_modelling_{}_with_category_as_sentence_explicit.txt'.format(dataset_name, split, dataset_name)
    # #     file_path = data_dir.joinpath(relative_path)
    # #     with open(file_path, 'w') as file:
    # #
    # #         for index, row in dataset[split].iterrows():
    # #             #preprocess value
    # #             prep_title = preprocess(row['title'])
    # #             categories = row['path_list'].split('>')
    # #             categories = [value.split('_')[1] for value in categories]
    # #             categories = [preprocess(value) for value in categories]
    # #             prep_catgories = ', '.join(categories)
    # #             line = 'The product {} belongs to the categories {}.\n'.format(prep_title, prep_catgories)
    # #             file.write(line)
    # #
    # # logger.info('Dataset prepared for language modelling {}!'.format(dataset_name))
    #
    #     relative_path = 'data/processed/{}/language-modelling/{}_language_modelling_{}_with_descriptions_additional_ds_{}.txt'.format(
    #                     dataset_name, split, dataset_name, suffix)
    #     file_path = data_dir.joinpath(relative_path)
    #     with open(file_path, 'w') as file:
    #
    #         for index, row in dataset[split].iterrows():
    #         # preprocess value
    #             prep_title = preprocess(row['title'])
    #
    #             description_values = row['description'].split('.')
    #             description = ''
    #             for value in description_values:
    #                 if len(value) > 4:
    #                     prep_value = preprocess(value)
    #                     description = '{} {}'.format(description,prep_value).strip()
    #
    #             line = '{} - {}\n'.format(prep_title, description)
    #             file.write(line)
    #
    #         if split == 'train':
    #             for index, row in df_additional.iterrows():
    #                 if type(row['Description']) is str:
    #                     description_values = row['Description'].split('.')
    #                     description = ''
    #                     for value in description_values:
    #                         if len(value) > 4:
    #                             prep_value = preprocess(value)
    #                             description = '{} - {}'.format(description, prep_value).strip()
    #
    #                     line = '{} - {}.\n'.format(row['Title'], description)
    #                     file.write(line)
    #
    #     logger.info('File {} created for Language Modelling!'.format(relative_path))
    #
    #     relative_path = 'data/processed/{}/language-modelling/{}_language_modelling_{}_with_category_as_sentence_and_descriptions_additional_ds_{}.txt'.format(
    #         dataset_name, split, dataset_name, suffix)
    #     file_path = data_dir.joinpath(relative_path)
    #     with open(file_path, 'w') as file:
    #
    #         for index, row in dataset[split].iterrows():
    #             # preprocess value
    #             prep_title = preprocess(row['title'])
    #             categories = row['path_list'].split('>')
    #             categories = [value.split('_')[1] for value in categories]
    #             categories = [preprocess(value) for value in categories]
    #             prep_catgories = ', '.join(categories)
    #             line = '{} belongs to {}.\n'.format(prep_title, prep_catgories)
    #             file.write(line)
    #
    #             description_values = row['description'].split('.')
    #             description = ''
    #             for value in description_values:
    #                 if len(value) > 4:
    #                     prep_value = preprocess(value)
    #                     description = '{} {}'.format(description, prep_value).strip()
    #
    #             line = '{} - {}.\n'.format(prep_title, description)
    #             file.write(line)
    #
    #         if split == 'train':
    #             for index, row in df_additional.iterrows():
    #                 categories = []
    #                 if row['Category'] is not None and type(row['Category']) is str:
    #                     categories.append(row['Category'])
    #                 if row['Breadcrumb'] is not None and type(row['Breadcrumb']) is str:
    #                     categories.append(row['Breadcrumb'])
    #                 if row['BreadcrumbList'] is not None and type(row['BreadcrumbList']) is str:
    #                     categories.append(row['BreadcrumbList'])
    #
    #                 if len(categories) > 0:
    #                     prep_categories = ' '.join(categories).strip()
    #
    #                     line = '{} belongs to {}.\n'.format(row['Title'], prep_categories)
    #                     file.write(line)
    #
    #                 if type(row['Description']) is str:
    #                     description_values = row['Description'].split('.')
    #                     description = ''
    #                     for value in description_values:
    #                         if len(value) > 4:
    #                             prep_value = preprocess(value)
    #                             description = '{}. {}'.format(description, prep_value).strip()
    #
    #                     line = '{} - {}\n'.format(row['Title'], description)
    #                     file.write(line)
    #
    #     logger.info('File {} created for Language Modelling!'.format(relative_path))

        relative_path = 'data/processed/{}/language-modelling/{}_language_modelling_{}_with_category_and_descriptions_one_line_additional_ds_{}.txt'.format(
            dataset_name, split, dataset_name, suffix)
        file_path = data_dir.joinpath(relative_path)
        with open(file_path, 'w') as file:

            for index, row in dataset[split].iterrows():
                # preprocess value
                prep_title = preprocess(row['title'])
                categories = row['path_list'].split('>')
                categories = [value.split('_')[1] for value in categories]
                categories = [preprocess(value) for value in categories]
                prep_catgories = ', '.join(categories)

                description_values = row['description'].split('.')
                description = ''
                for value in description_values:
                    if len(value) > 4:
                        prep_value = preprocess(value)
                        description = '{} {}'.format(description, prep_value).strip()

                line = '{} - {} - {}.\n'.format(prep_title, prep_catgories, description)
                file.write(line)

            if split == 'train':
                for index, row in df_additional.iterrows():
                    categories = []
                    if row['Category'] is not None and type(row['Category']) is str:
                        categories.append(row['Category'])
                    if row['Breadcrumb'] is not None and type(row['Breadcrumb']) is str:
                        categories.append(row['Breadcrumb'])
                    if row['BreadcrumbList'] is not None and type(row['BreadcrumbList']) is str:
                        categories.append(row['BreadcrumbList'])

                    line = row['Title']
                    if len(categories) > 0:
                        prep_categories = ' '.join(categories).strip()

                        line = '{} - {}'.format(line, prep_categories)

                    if type(row['Description']) is str:
                        description_values = row['Description'].split('.')
                        description = ''
                        for value in description_values:
                            if len(value) > 4:
                                prep_value = preprocess(value)
                                description = '{} {}'.format(description, prep_value).strip()

                        line = '{} - {}'.format(line, description)
                    file.write('{}\n'.format(line))

        logger.info('File {} created for Language Modelling!'.format(relative_path))

        relative_path = 'data/processed/{}/language-modelling/{}_language_modelling_{}_with_category_reverse_and_descriptions_one_line_additional_ds_{}.txt'.format(
            dataset_name, split, dataset_name, suffix)
        file_path = data_dir.joinpath(relative_path)
        with open(file_path, 'w') as file:

            for index, row in dataset[split].iterrows():
                # preprocess value
                prep_title = preprocess(row['title'])
                categories = row['path_list'].split('>')
                categories = [value.split('_')[1] for value in categories]
                categories = [preprocess(value) for value in categories]
                categories.reverse()
                prep_catgories = ', '.join(categories)

                description_values = row['description'].split('.')
                description = ''
                for value in description_values:
                    if len(value) > 4:
                        prep_value = preprocess(value)
                        description = '{} {}'.format(description, prep_value).strip()

                line = '{} - {} - {}.\n'.format(prep_title, prep_catgories, description)
                file.write(line)

            if split == 'train':
                for index, row in df_additional.iterrows():
                    categories = []
                    if row['Category'] is not None and type(row['Category']) is str:
                        categories.append(row['Category'])
                    if row['Breadcrumb'] is not None and type(row['Breadcrumb']) is str:
                        categories.append(row['Breadcrumb'])
                    if row['BreadcrumbList'] is not None and type(row['BreadcrumbList']) is str:
                        categories.append(row['BreadcrumbList'])

                    line = row['Title']
                    if len(categories) > 0:
                        categories.reverse()
                        prep_categories = ' '.join(categories).strip()

                        line = '{} - {}'.format(line, prep_categories)

                    if type(row['Description']) is str:
                        description_values = row['Description'].split('.')
                        description = ''
                        for value in description_values:
                            if len(value) > 4:
                                prep_value = preprocess(value)
                                description = '{} {}'.format(description, prep_value).strip()

                        line = '{} - {}'.format(line, description)
                    file.write('{}\n'.format(line))

        logger.info('File {} created for Language Modelling!'.format(relative_path))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()