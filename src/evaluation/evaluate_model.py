#!/usr/bin/env python3

import logging

import click

from src.evaluation.evaluator.model_evaluator_fasttext import ModelEvaluatorFastText
from src.evaluation.evaluator.model_evaluator_random_forest import ModelEvaluatorRandomForest
from src.evaluation.evaluator.model_evaluator_transformer_flat import ModelEvaluatorTransformer


@click.command()
@click.option('--configuration', help='Path to configuration')
@click.option('--test/--no-test', default=False, help='Test configuration - Run only on small subset')
@click.option('--experiment_type', help='Experiment Type')
def main(configuration, test, experiment_type):
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    if experiment_type == 'eval-transformer-based':
        evaluator = ModelEvaluatorTransformer(configuration, test, experiment_type)
    elif experiment_type == 'eval-random-forest-based':
        evaluator = ModelEvaluatorRandomForest(configuration, test, experiment_type)
    elif experiment_type == 'eval-fasttext-based':
        evaluator = ModelEvaluatorFastText(configuration, test, experiment_type)
    else:
        raise ValueError('Experiment Type {} not defined!'.format(experiment_type))

    evaluator.evaluate()


if __name__ == '__main__':
    main()
