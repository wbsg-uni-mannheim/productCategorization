import logging

import click

from src.data.make_dataset import load_line_json

@click.command()
@click.option('--file_path', help='Path to file containing products')
@click.option('--output_path', help='Path to output file')
def main(file_path, output_path):
    df_data = load_line_json(file_path)
    hosts = df_data['URL'].values
    hosts = set([extract_host(host) for host in hosts])

    with open(output_path, 'w') as out_file:
        for host in hosts:
            out_file.write('{}\n'.format(host))

def extract_host(value):
    value = value.replace('https://', '')
    value = value.replace('http://', '')
    value = value.split('/')[0]
    value = value.replace('www.', '')

    return value


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()