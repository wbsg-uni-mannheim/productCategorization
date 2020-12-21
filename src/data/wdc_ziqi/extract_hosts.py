import logging

import click

from src.data.make_dataset import load_line_json

@click.command()
@click.option('--file_paths', help='Path to file containing products', multiple=True)
@click.option('--output_path', help='Path to output file')
def main(file_paths, output_path):
    hosts = set()

    for file_path in file_paths:
        df_data = load_line_json(file_path)
        new_hosts = df_data['URL'].values
        for host in new_hosts:
            hosts.add(extract_host(host))

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