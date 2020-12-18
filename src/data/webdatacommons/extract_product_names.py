import csv
import gzip
import logging
import copy
from multiprocessing import Process
import time

import click

from src.data.preprocessing import preprocess


@click.command()
@click.option('--file_path', help='Path to file containing products')
@click.option('--output_path', help='Path to file containing products')
def main(file_path, output_path):
    logger = logging.getLogger(__name__)

    categories = set()
    breadcrumbs = set()
    breadcrumbLists = set()

    collected_products = []
    counter = 0
    product = {'Title': 'Title', 'Description': 'Description', 'Category': 'Category',
               'Breadcrumb': 'Breadcrumb', 'BreadcrumbList': 'BreadcrumbList',
               'Breadcrumb-Predicate': 'Breadcrumb-Predicate'}
    uri = 'initial'
    node = ''
    node_relevant = False
    p = None  # Process for Multithreading
    print_next_values = 0

    # Initialize output file
    open(output_path, 'w').close()
    logger.info('Inialize output file {}!'.format(output_path))

    with gzip.open(file_path, 'rt', encoding='utf-8') as f:

        for i, line in enumerate(f):
            reader = csv.reader([line], delimiter=' ', quotechar='"')
            try:
                for r in reader:
                    if len(r) > 4:

                        if r[0] != node:
                            node_relevant = False
                        #if node_relevant:
                            #logger.info(r)
                        if print_next_values > 0:
                            print_next_values -= 1
                            #logger.info(r)

                        if r[3] != uri:
                            uri = r[3]
                            if len(product['Title']) > 0 and (len(product['Category']) > 0 or
                                                              len(product['Breadcrumb']) > 0 or
                                                              len(product['BreadcrumbList']) > 0):

                                collected_products.append(copy.deepcopy(product))
                                # Initialize product dict
                                product = {key: '' for key in product}
                                counter += 1

                                if counter % 10000 == 0:
                                    p = parallel_write(p, collected_products, output_path)
                                    logger.info('Written {} product names to disc.'.format(counter))

                                    for value in categories:
                                        logger.info('Category value: {}'.format(value))

                                    for value in breadcrumbs:
                                        logger.info('Breadcrumbs value: {}'.format(value))

                                    for value in breadcrumbLists:
                                        logger.info('Breadcrumblists value: {}'.format(value))

                        if r[0] == '_:nodefe8e433a782f383d89dc215c26b12724':
                            logger.info(r)

                        if r[1] == '<http://schema.org/Product/name>' and '@en' in r[2]:
                            prep_value = preprocess_value(r[2])
                            if len(prep_value) > 0 and prep_value != 'null':
                                product['Title'] = prep_value

                        if r[1] == '<http://schema.org/Product/description>':
                            prep_value = preprocess_value(r[2])
                            if len(prep_value) > 0 and prep_value != 'null':
                                product['Description'] = prep_value

                        if 'breadcrumblist' in r[2].lower():
                            node = r[0]
                            node_relevant = True
                            #logger.info(r)

                        if r[1] == '<http://schema.org/Product/breadcrumb>':
                            if '_:node' in r[2]:
                                node = r[2]
                                node_relevant = True
                                logger.info(r)
                                print_next_values = 5
                            else:
                                prep_value = preprocess_value(r[2])
                                if len(prep_value) > 0 and prep_value != 'null':
                                    product['Breadcrumb'] = prep_value

                        if 'category' in r[1].lower():
                            prep_value = preprocess_value(r[2])
                            if len(prep_value) > 0 and prep_value != 'null':
                                product['Category'] = '{} {}'.format(product['Category'], prep_value).lstrip()
                                categories.add(r[1])

                        if 'breadcrumblist' in r[1].lower():
                            prep_value = preprocess_value(r[2])
                            if len(prep_value) > 0 and prep_value != 'null':
                                product['BreadcrumbList'] = '{} {}'.format(product['BreadcrumbList'],
                                                                           prep_value).lstrip()
                                breadcrumbLists.add(r[1])

                        elif 'breadcrumb' in r[1].lower():
                            prep_value = preprocess_value(r[2])
                            if len(prep_value) > 0 and prep_value != 'null':
                                product['Breadcrumb'] = '{} {}'.format(product['Breadcrumb'], prep_value).lstrip()
                                product['Breadcrumb-Predicate'] = '{} {}'.format(product['Breadcrumb-Predicate'],
                                                                                 r[1]).lstrip()
                                breadcrumbs.add(r[1])

            except csv.Error as e:
                print(e)

    p = parallel_write(p, collected_products, output_path)
    logger.info('Written {} product names to disc.'.format(counter))
    p.join()

    for value in categories:
        logger.info('Category value: {}'.format(value))

    for value in breadcrumbs:
        logger.info('Breadcrumbs value: {}'.format(value))

    for value in breadcrumbLists:
        logger.info('Breadcrumblists value: {}'.format(value))


def parallel_write(p, products, path):
    logger = logging.getLogger(__name__)
    if p is not None:
        start = time.time()
        p.join()
        end = time.time()
        elapsed_time = end - start
        logger.info('Waited for {}'.format(elapsed_time))
    p = Process(target=write_to_disk, args=(copy.deepcopy(products), path))
    p.start()
    return p


def write_to_disk(products, path):
    with open(path, 'a') as out_f:
        for product in products:
            line = '{};{};{};{};{};{}\n'.format(product['Title'], product['Description'],
                                                product['Category'], product['Breadcrumb'],
                                                product['BreadcrumbList'],
                                                product['Breadcrumb-Predicate'])
            out_f.write(line)


def preprocess_value(value):
    value = value.split('@')[0]
    value = value.replace('\\n', '').replace('\\t', '').replace("\xc2\xa0", " ")
    prep_value = preprocess(value)
    return prep_value


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
