#!/usr/bin/env python3

import logging

import click

from src.experiments.experiment_runner import ExperimentRunner


@click.command()
@click.option('--configuration', help='Configuration used to run the experiments')
@click.option('--test/--no-test', default=False, help='Test configuration - Run only on small subset')
def main(configuration, test):
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    runner = ExperimentRunner(configuration, test)
    runner.run()


if __name__ == '__main__':
    main()
