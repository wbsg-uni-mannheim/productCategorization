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
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        with open(output_path, 'w', encoding='utf-8') as out_f:
            counter = 0
            for i, line in enumerate(f):
                reader = csv.reader([line], delimiter= ' ', quotechar='"')
                try:
                    for r in reader:
                        if r[1] == '<http://schema.org/Product/name>' and '@en' in r[2]:

                            title = r[2].split('@')[0]
                            prep_title = preprocess(title)
                            line = '{} \n'.format(prep_title)
                            out_f.write(line)
                            counter += 1

                            if counter % 100 == 0:
                                logger.info('Written {} product names to disc.'.format(counter))

                            if counter == 100000:
                                break

                except csv.Error as e:
                    print(e)

    logger.info('Written {} product names to disc.'.format(counter))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()