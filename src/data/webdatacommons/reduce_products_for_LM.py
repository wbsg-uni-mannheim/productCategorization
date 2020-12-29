import logging

import pandas as pd
import click
from tqdm import tqdm


@click.command()
@click.option('--file', help='Input file')
@click.option('--host_path', help='Input file')
@click.option('--output_file', help='Output file')
@click.option('--num_products_per_host', help='Number of extracted products per host', type=int)
def main(file, host_path, output_file, num_products_per_host):
    logger = logging.getLogger(__name__)

    list_products = []
    df_products = pd.read_csv(file, sep=';')
    logger.info('{} Products loaded!'.format(len(df_products)))
    hosts = load_hosts(host_path)

    for host in tqdm(hosts):
        df_sub_products = df_products[df_products['host'].str.contains(host)].sample(frac=1)
        df_sub_products = df_sub_products.head(num_products_per_host)
        list_products.append(df_sub_products)

    df_reduced_products = pd.concat(list_products)
    df_reduced_products.to_csv(output_file, sep=';', index=False)
    logger.info('Results written to {}!'.format(output_file))

    logger.info('Added products {}!'.format(len(df_reduced_products)))
    logger.info('Average number of added products {}!'.format(len(df_reduced_products)/ len(hosts)))

def load_hosts(host_path):
    logger = logging.getLogger(__name__)
    hosts = []
    counter = 0
    with open(host_path, 'r') as host_file:
        lines = host_file.readlines()
        for line in lines:
            hosts.append(line.strip())
            counter += 1

    logger.info('Loaded {} hosts!'.format(counter))
    return hosts

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()