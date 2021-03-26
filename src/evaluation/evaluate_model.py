#!/usr/bin/env python3
import json
import logging
from sys import platform

import click

from src.evaluation.evaluator.model_evaluator_fasttext import ModelEvaluatorFastText
from src.evaluation.evaluator.model_evaluator_random_forest import ModelEvaluatorRandomForest
from src.evaluation.evaluator.model_evaluator_transformer_flat import ModelEvaluatorTransformer
from src.evaluation.evaluator.model_evaluator_transformer_rnn import ModelEvaluatorTransformerRNN
from src.evaluation.evaluator.model_evaluator_transformer_hierarchy import ModelEvaluatorTransformerHierarchy


@click.command()
@click.option('--configuration', help='Configuration used to run the experiments')
@click.option('--test/--no-test', default=False, help='Test configuration - Run only on small subset')
@click.option('--experiment_type', help='Experiment Type')
def main(configuration, test, experiment_type):
    logger = logging.getLogger(__name__)
    #Extract experiment name from config for logging
    if platform == "win32":
        config_path = configuration.split('\\')
    else:
        config_path = configuration.split('/')
    dataset = config_path[-2]
    experiment_name = config_path[-1].split('.')[0]

    log_file = '{}_{}.log'.format(dataset, experiment_name)

    #Activate Logging
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO, format=log_fmt)

    # Augment experiments
    generated_experiments = augment_experiments(configuration)

    if test:
        generated_experiments = generated_experiments[:1]

    for config in generated_experiments:

        evaluate_experiments(config, test, experiment_type)

    logger.info('Finished running experiments from {}!'.format(configuration))

#@click.command()
#@click.option('--configuration', help='Path to configuration')
#@click.option('--test/--no-test', default=False, help='Test configuration - Run only on small subset')
#@click.option('--experiment_type', help='Experiment Type')
def evaluate_experiments(configuration, test, experiment_type):
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    if experiment_type == 'eval-transformer-based':
        evaluator = ModelEvaluatorTransformer(configuration, test, experiment_type)
    elif experiment_type == 'eval-transformer-based-rnn':
        evaluator = ModelEvaluatorTransformerRNN(configuration, test, experiment_type)
    elif experiment_type == 'eval-transformer-based-hierarchy':
        evaluator = ModelEvaluatorTransformerHierarchy(configuration, test, experiment_type)
    elif experiment_type == 'eval-random-forest-based':
        evaluator = ModelEvaluatorRandomForest(configuration, test, experiment_type)
    elif experiment_type == 'eval-fasttext-based':
        evaluator = ModelEvaluatorFastText(configuration, test, experiment_type)
    else:
        raise ValueError('Experiment Type {} not defined!'.format(experiment_type))

    evaluator.evaluate()

def augment_experiments(experiment):
    logger = logging.getLogger(__name__)

    learning_rates = [5e-5]
    seeds = [42, 13, 9]

    # Load base configuration
    with open(experiment) as f:
        exp_config = json.load(f)

    experiment_path = experiment.split('.')[0]
    initial_prediction_output = exp_config['prediction_output']
    initial_model_path = exp_config['model_path']

    # Generate new configurations
    generated_experiments = []
    for learning_rate in learning_rates:
        exp_config['learning_rate'] = learning_rate
        for seed in seeds:
            exp_config['seed'] = seed
            if platform == "win32":
                prefix_experiment_name = experiment_path.split('\\')[-1]
            else:
                prefix_experiment_name = experiment_path.split('/')[-1]
            experiment_name = '{}_{}_{}'.format(prefix_experiment_name, learning_rate, seed)
            exp_config['experiment_name'] = experiment_name
            prediction_output = '{}_{}_{}.csv'.format(initial_prediction_output, learning_rate, seed)
            exp_config['prediction_output'] = prediction_output
            model_path = '{}_{}_{}'.format(initial_model_path, learning_rate, seed)
            exp_config['model_path'] = model_path


            path_new_config = '{}_{}_{}.json'.format(experiment_path, learning_rate, seed)
            with open(path_new_config, 'w') as json_file:
                json.dump(exp_config, json_file)
                logger.info('New configuration generated and saved at {}!'.format(path_new_config))

            generated_experiments.append(path_new_config)

    return generated_experiments

if __name__ == '__main__':
    main()
