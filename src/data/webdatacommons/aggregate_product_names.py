import logging
from os import listdir
import click
import pandas as pd
from tqdm import tqdm

@click.command()
@click.option('--file_dir', help='Path to dir containing files with products')
@click.option('--output_file', help='Path to output_dir')
def main(file_dir, output_file):
    logger = logging.getLogger(__name__)

    logger.info('Start to aggregate products')

    list_dataframes = []

    for file in tqdm(listdir(file_dir)):
        if '.txt' in file:
            file_path = '{}/{}'.format(file_dir, file)
            try:
                df_new_products = pd.read_csv(filepath_or_buffer=file_path, sep=';', error_bad_lines=False)
                df_new_products = drop_duplicates(df_new_products)
                df_new_products = remove_hosts_based_on_count(df_new_products, 100)


                list_dataframes.append(df_new_products)

            except pd.errors.EmptyDataError:
                logger.info('File {} is empty!'.format(file_path))

    logger.info('Concat dataframes!')
    df_products = pd.concat(list_dataframes)
    df_products = drop_duplicates(df_products)
    df_products = remove_hosts_based_on_count(df_products, 100)
    df_products.to_csv(output_file, sep=';', index=False)
    logger.info('Aggregated results written to {}!'.format(output_file))
    logger.info('Aggregated dataset contains {} products!'.format(len(df_products)))

def drop_duplicates(df):
    df.sort_values(by=['Category', 'Breadcrumb', 'BreadcrumbList', 'Description'], inplace=True)
    df.drop_duplicates(subset=['Title'], inplace=True)

    return df

def remove_hosts_based_on_count(df, count):
    # Remove hosts based on count
    host_counts = df['Host'].value_counts()
    for host, counts in host_counts[host_counts > count].items():
        # Shuffle and choose rows to be dropped
        df_products_to_be_dropped = df[df['Host'] == host].sample(frac=1)[count:]
        df.drop(df_products_to_be_dropped.index, inplace=True)

    return df

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()