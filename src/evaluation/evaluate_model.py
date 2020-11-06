#!/usr/bin/env python3

import logging

import click

from src.evaluation.model_evaluator import ModelEvaluator


@click.command()
@click.option('--configuration', help='Path to configuration')
def main(configuration):
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    evaluator = ModelEvaluator(configuration)
    evaluator.evaluate()


if __name__ == '__main__':
    main()