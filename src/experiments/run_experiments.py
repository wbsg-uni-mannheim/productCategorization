#!/usr/bin/env python3

import logging

import click

from src.experiments.experiment_runner import ExperimentRunner
from src.experiments.experiment_runner_dict import ExperimentRunnerDict
from src.experiments.experiment_runner_transformer import ExperimentRunnerTransformer


@click.command()
@click.option('--configuration', help='Configuration used to run the experiments')
@click.option('--test/--no-test', default=False, help='Test configuration - Run only on small subset')
@click.option('--experiment_type', default=False, help='Experiment Type')
def main(configuration, test, experiment_type):
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    if experiment_type == 'dict-based':
        runner = ExperimentRunnerDict(configuration, test, experiment_type)
    elif experiment_type == 'transformer-based':
        runner = ExperimentRunnerTransformer(configuration, test, experiment_type)
    else:
        raise ValueError('Experiment Type {} not defined!'.format(experiment_type))
    runner.run()


if __name__ == '__main__':
    main()
