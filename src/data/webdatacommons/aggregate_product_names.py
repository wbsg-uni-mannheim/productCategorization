import logging
from os import listdir
import click
import pandas as pd

@click.command()
@click.option('--file_dir', help='Path to dir containing files with products')
@click.option('--output_file', help='Path to output_dir')
def main(file_dir, output_file):
    logger = logging.getLogger(__name__)

    logger.info('Start to aggregate products')

    df_products = pd.DataFrame()

    for file in listdir(file_dir):
        if '.txt' in file:
            file_path = '{}/{}'.format(file_dir, file)
            df_new_products = pd.read_csv(filepath_or_buffer=file_path, sep=';')
            df_new_products = drop_duplicates(df_new_products)

            # Drop duplicates from complete df
            df_products = df_products.append(df_new_products, ignore_index=True)
            df_products = drop_duplicates(df_products)

    df_products.to_csv(output_file, sep=';', index=False)
    logger.info('Aggregated results written to {}!'.format(output_file))

def drop_duplicates(df):
    df.sort_values(by=['Category', 'Breadcrumb', 'BreadcrumbList', 'Description'])
    df.drop_duplicates(subset=['Title'], inplace=True)

    return df

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()