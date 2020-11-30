import json
import os
import unittest
from pathlib import Path


class TestConfiguration(unittest.TestCase):

    def test_unique_configurations(self):
        """Proof configuration to be valid!"""
        datasets = ['icecat', 'webdatacommons', 'rakuten', 'wdc_ziqi']
        project_dir = Path(__file__).resolve().parents[3]

        dataset_experiment_name_combination = {}
        original_experiment_dataset_names = []

        for dataset_name in datasets:
            relative_path = 'experiments/{}/'.format(dataset_name)
            absolute_path = project_dir.joinpath(relative_path)

            # Load all configurations
            prediction_outputs = []
            experiment_names = []
            parameter_hashes = []

            configurations = os.listdir(absolute_path)
            n_configurations = len(configurations)
            counter = 0
            for filename in configurations:
                file_path = absolute_path.joinpath(filename)
                with open(file_path) as json_file:
                    experiments = json.load(json_file)
                    ds_name = experiments['dataset']

                    msg = 'Dataset not correctly maintained for coniguration in {}!'.format(file_path)
                    self.assertEqual(dataset_name, ds_name, msg)

                    experiment_type = experiments["type"]
                    new_model_experiment_types = ['transformer-based', 'transformer-based-hierarchy', 'transformer-based-rnn',
                                                  'random-forest-based', 'fasttext-based']
                    eval_experiment_types = ['eval-transformer-based', 'eval-random-forest-based', 'eval-fasttext-based']

                    huggingface_experiment_types = ['language-modelling']

                    if experiment_type in new_model_experiment_types:
                        counter = counter + 1
                        parameter = experiments["parameter"]
                        hash_parameter = hash(str(parameter))
                        self.assertNotIn(hash_parameter, parameter_hashes, 'Parameter of configuration {} already known!'
                                         .format(file_path))
                        parameter_hashes.append(hash_parameter)

                        experiment_name = parameter["experiment_name"]

                    elif experiment_type in huggingface_experiment_types:
                        counter = counter + 1
                        experiment_name = experiments["experiment_name"]

                    elif experiment_type in eval_experiment_types:
                        counter = counter + 1
                        experiment_name = experiments["experiment_name"]

                    elif experiment_type == 'dict-based':
                        counter = counter + 1
                        # Add dummy experiment name - Only one dict based config per dataset!
                        experiment_name = 'dictionary_based_models'
                    else:
                        raise ValueError('Experiment Type {} not maintained!'.format(experiment_type))

                    # Check duplicate experiment names!
                    self.assertNotIn(experiment_name, experiment_names, 'Duplicated experiment name: {} - Hint: file: {}'
                                     .format(experiment_name, file_path))

                    # Check experiment name matches name of configuration file!
                    derived_filename = '{}.json'.format(experiment_name)
                    msg = 'Configuration {} does not match experiment name {}!'.format(filename, derived_filename)
                    self.assertEqual(filename, derived_filename, msg=msg)

                    if 'eval-' in experiment_type:
                        # Evaluate if original experiment exists
                        original_experiment_name = experiments["original_experiment_name"]
                        original_dataset = experiments["original_dataset"]

                        experiment_reference = (original_dataset, original_experiment_name, experiment_name)
                        original_experiment_dataset_names.append(experiment_reference)

                        prediction_output = experiments["prediction_output"]
                        self.assertNotIn(prediction_output, prediction_outputs, 'Duplicated prediction output: {}'
                                     .format(prediction_output))
                        prediction_outputs.append(prediction_output)

                    experiment_names.append(experiment_name)

                dataset_experiment_name_combination[dataset_name] = experiment_names.copy()

            for original_dataset, original_experiment_name, experiment_name in original_experiment_dataset_names:
                self.assertIn(original_experiment_name, dataset_experiment_name_combination[original_dataset],
                              'Original experiment of evaluation experiment {} does not exist!'. format(experiment_name))

            self.assertEqual(n_configurations, counter,
                             'Some configurations of dataset {} have an unknown experiment type!'.format(dataset_name))


if __name__ == '__main__':
    unittest.main()
