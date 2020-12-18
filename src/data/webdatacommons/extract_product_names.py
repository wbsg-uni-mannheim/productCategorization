import csv
import gzip
import logging

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

    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        with open(output_path, 'w', encoding='utf-8') as out_f:
            counter = 0
            product = {'Title': 'Title', 'Description': 'Description', 'Category': 'Category',
                       'Breadcrumb': 'Breadcrumb', 'BreadcrumbList': 'BreadcrumbList',
                       'Breadcrumb-Predicate': 'Breadcrumb-Predicate'}
            uri = 'initial'
            node = ''
            node_relevant = False
            for i, line in enumerate(f):
                reader = csv.reader([line], delimiter= ' ', quotechar='"')
                try:
                    for r in reader:
                        if len(r) > 4:

                            if r[0] != node:
                                node_relevant = False
                            if node_relevant:
                                logger.info(r)

                            if r[3] != uri:
                                uri = r[3]
                                if len(product['Title']) > 0 and \
                                    (len(product['Category']) > 0 \
                                    or len(product['Breadcrumb']) > 0 \
                                    or len(product['BreadcrumbList']) > 0):

                                    line = '{};{};{};{};{};{}\n'.format(product['Title'], product['Description'],
                                                              product['Category'], product['Breadcrumb'],
                                                              product['BreadcrumbList'],
                                                              product['Breadcrumb-Predicate'])
                                    out_f.write(line)
                                    # Initialize product dict
                                    product = {key: '' for key in product }
                                    counter += 1

                                    if counter % 10000 == 0:
                                        logger.info('Written {} product names to disc.'.format(counter))

                                        for value in categories:
                                            logger.info('Category value: {}'.format(value))

                                        for value in breadcrumbs:
                                            logger.info('Breadcrumbs value: {}'.format(value))

                                        for value in breadcrumbLists:
                                            logger.info('Breadcrumblists value: {}'.format(value))

                                    if counter == 10000:
                                        break

                            if r[1] == '<http://schema.org/Product/name>' and '@en' in r[2]:
                                prep_value = preprocess_value(r[2])
                                if len(prep_value) > 0 and prep_value != 'null':
                                    product['Title'] = prep_value

                            if r[1] == '<http://schema.org/Product/description>':
                                prep_value = preprocess_value(r[2])
                                if len(prep_value) > 0 and prep_value != 'null':
                                    product['Description'] = prep_value

                            if r[1] == '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>' \
                                    and 'breadcrumblist' in r[2].lower():
                                node = r[0]
                                node_relevant = True
                                logger.info(r)

                            if 'category' in r[1].lower():
                                prep_value = preprocess_value(r[2])
                                if len(prep_value) > 0 and prep_value != 'null':
                                    product['Category'] = '{} {}'.format(product['Category'], prep_value).lstrip()
                                    categories.add(r[1])

                            if 'breadcrumblist' in r[1].lower():
                                prep_value = preprocess_value(r[2])
                                if len(prep_value) > 0 and prep_value != 'null':
                                    product['BreadcrumbList'] = '{} {}'.format(product['BreadcrumbList'], prep_value).lstrip()
                                    breadcrumbLists.add(r[1])

                            elif 'breadcrumb' in r[1].lower():
                                prep_value = preprocess_value(r[2])
                                if len(prep_value) > 0 and prep_value != 'null':
                                    product['Breadcrumb'] = '{} {}'.format(product['Breadcrumb'], prep_value).lstrip()
                                    product['Breadcrumb-Predicate'] = '{} {}'.format(product['Breadcrumb-Predicate'], r[1]).lstrip()
                                    breadcrumbs.add(r[1])

                except csv.Error as e:
                    print(e)

    logger.info('Written {} product names to disc.'.format(counter))

    for value in categories:
        logger.info('Category value: {}'.format(value))

    for value in breadcrumbs:
        logger.info('Breadcrumbs value: {}'.format(value))

    for value in breadcrumbLists:
        logger.info('Breadcrumblists value: {}'.format(value))

def preprocess_value(value):
    prep_value = preprocess(value.split('@')[0].replace('\\n', ''))
    return prep_value

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()