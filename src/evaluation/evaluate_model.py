#!/usr/bin/env python3

import logging

import click

from src.evaluation.model_evaluator import ModelEvaluator


@click.command()
@click.option('--configuration', help='Path to configuration')
@click.option('--test/--no-test', default=False, help='Test configuration - Run only on small subset')
def main(configuration, test):
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    evaluator = ModelEvaluator(configuration, test)
    evaluator.evaluate()


if __name__ == '__main__':
    main()