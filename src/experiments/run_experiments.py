#!/usr/bin/env python3

import click
import logging
#import torch.multiprocessing as mp
import json

from src.experiments.runner.experiment_runner_dict import ExperimentRunnerDict
from src.experiments.runner.experiment_runner_fasttext import ExperimentRunnerFastText
from src.experiments.runner.experiment_runner_random_forest import ExperimentRunnerRandomForest
from src.experiments.runner.experiment_runner_transformer_att_rnn import ExperimentRunnerTransformerAttRNN
from src.experiments.runner.experiment_runner_transformer_flat import ExperimentRunnerTransformerFlat
from src.experiments.runner.experiment_runner_transformer_rnn import ExperimentRunnerTransformerRNN
from src.experiments.runner.experiment_runner_transformer_hierarchy import ExperimentRunnerTransformerHierarchy

@click.command()
@click.option('--configuration', help='Configuration used to run the experiments')
@click.option('--test/--no-test', default=False, help='Test configuration - Run only on small subset')
@click.option('--experiment_type', help='Experiment Type')
def main(configuration, test, experiment_type):
    #Extract experiment name from config for logging
    config_path = configuration.split('/')
    dataset = config_path[-2]
    experiment_name = config_path[-1].split('.')[0]

    log_file = '{}_{}.log'.format(dataset, experiment_name)

    #Activate Logging
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO, format=log_fmt)

    # Augment experiments
    generated_experiments = augment_experiments(configuration)

    for config in generated_experiments:
        run_experiment(config, test, experiment_type)

        #process = mp.Process(target=run_experiment, args=args)
        #process.start()
        #process.join()

#@click.command()
#@click.option('--configuration', help='Configuration used to run the experiments')
#@click.option('--test/--no-test', default=False, help='Test configuration - Run only on small subset')
#@click.option('--experiment_type', help='Experiment Type')
def run_experiment(configuration, test, experiment_type):

    if experiment_type == 'dict-based':
        runner = ExperimentRunnerDict(configuration, test, experiment_type)
    elif experiment_type == 'transformer-based':
        runner = ExperimentRunnerTransformerFlat(configuration, test, experiment_type)
    elif experiment_type == 'transformer-based-rnn':
        runner = ExperimentRunnerTransformerRNN(configuration, test, experiment_type)
    elif experiment_type == 'transformer-based-att-rnn':
        runner = ExperimentRunnerTransformerAttRNN(configuration, test, experiment_type)
    elif experiment_type == 'transformer-based-hierarchy':
        runner = ExperimentRunnerTransformerHierarchy(configuration, test, experiment_type)
    elif experiment_type == 'random-forest-based':
        runner = ExperimentRunnerRandomForest(configuration, test, experiment_type)
    elif experiment_type == 'fasttext-based':
        runner = ExperimentRunnerFastText(configuration, test, experiment_type)
    else:
        raise ValueError('Experiment Type {} not defined!'.format(experiment_type))
    runner.run()


def augment_experiments(experiment):
    logger = logging.getLogger(__name__)

    learning_rates = [1e-5, 5e-5, 1e-4]
    seeds = [42, 13, 9]

    # Load base configuration
    with open(experiment) as f:
        exp_config = json.load(f)

    experiment_path = experiment.split('.')[0]

    # Generate new configurations
    generated_experiments = []
    for learning_rate in learning_rates:
        exp_config['parameter']['learning_rate'] = learning_rate
        for seed in seeds:
            exp_config['parameter']['seed'] = seed
            prefix_experiment_name = experiment_path.split('/')[-1]
            experiment_name = '{}_{}_{}'.format(prefix_experiment_name, learning_rate, seed)
            exp_config['parameter']['experiment_name'] = experiment_name

            path_new_config = '{}_{}_{}.json'.format(experiment_path, learning_rate, seed)
            with open(path_new_config, 'w') as json_file:
                json.dump(exp_config, json_file)
                logger.info('New configuration generated and saved at {}!'.format(path_new_config))

            generated_experiments.append(path_new_config)

    return generated_experiments

if __name__ == '__main__':
    main()
