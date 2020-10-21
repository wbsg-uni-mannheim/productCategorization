import logging
import sys

import click

from src.experiments.experiment_runner import ExperimentRunner


@click.command()
@click.option('--path', help='Dataset which you like to preprocess')
def main(path):
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    runner = ExperimentRunner(path)
    runner.run()


if __name__ == '__main__':
    main()
