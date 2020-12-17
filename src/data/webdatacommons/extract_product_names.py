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
    breadcrumblists = set()

    with open(file_path, 'rt', encoding='utf-8') as f:
        with open(output_path, 'w', encoding='utf-8') as out_f:
            counter = 0
            product = {'Title': 'Title', 'Description': 'Description', 'Category': 'Category',
                       'Breadcrumb': 'Breadcrumb', 'BreadcrumbList': 'BreadcrumbList'}
            uri = 'initial'
            for i, line in enumerate(f):
                reader = csv.reader([line], delimiter= ' ', quotechar='"')
                try:
                    for r in reader:
                        if len(r) > 4:
                            if r[3] != uri:
                                uri = r[3]
                                if len(product['Title']) > 0 and \
                                    (len(product['Category']) > 0 \
                                    or len(product['Breadcrumb']) > 0 \
                                    or len(product['BreadcrumbList']) > 0):

                                    line = '{};{};{};{};{}\n'.format(product['Title'], product['Description'],
                                                              product['Category'], product['Breadcrumb'],
                                                              product['BreadcrumbList'])
                                    out_f.write(line)
                                    # Initialize product dict
                                    product = {key: '' for key in product }
                                    counter += 1

                                    if counter % 100 == 0:
                                        logger.info('Written {} product names to disc.'.format(counter))

                                    if counter == 100000:
                                        break

                            if r[1] == '<http://schema.org/Product/name>' and '@en' in r[2]:
                                product['Title'] = preprocess(r[2].split('@')[0].replace('\\n', ''))

                            if r[1] == '<http://schema.org/Product/description>':
                                product['Description'] = preprocess(r[2].split('@')[0].replace('\\n', ''))

                            if 'category' in r[1]:
                                categories.add(r[1])
                                product['Category'] = preprocess(r[2].split('@')[0].replace('\\n', ''))

                            if 'breadcrumb' in r[1]:
                                breadcrumbs.add(r[1])
                                product['Breadcrumb'] = preprocess(r[2].split('@')[0].replace('\\n', ''))

                            if 'breadcrumblist' in r[1]:
                                breadcrumblists.add(r[1])
                                product['BreadcrumbList'] = preprocess(r[2].split('@')[0].replace('\\n', ''))

                except csv.Error as e:
                    print(e)

    for value in categories:
        logger.info('Category value: {}'.format(value))

    for value in breadcrumbs:
        logger.info('Breadcrumbs value: {}'.format(value))

    for value in breadcrumblists:
        logger.info('Breadcrumblists value: {}'.format(value))

    logger.info('Written {} product names to disc.'.format(counter))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()