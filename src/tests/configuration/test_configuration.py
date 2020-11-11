import json
import os
import unittest
from pathlib import Path


class TestConfiguration(unittest.TestCase):

    def test_unique_configurations(self):
        """Proof configuration to be valid!"""
        datasets = ['icecat', 'webdatacommons', 'rakuten']
        project_dir = Path(__file__).resolve().parents[3]

        dataset_experiment_name_combination = {}
        original_experiment_dataset_names = []

        for dataset_name in datasets:
            relative_path = 'experiments/{}/configuration/'.format(dataset_name)
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
                    if experiment_type == 'transformer-based':
                        counter = counter + 1
                        parameter = experiments["parameter"][0] #Only first element of parameter is filled for now
                        hash_parameter = hash(str(parameter))
                        self.assertNotIn(hash_parameter, parameter_hashes, 'Parameter of configuration {} already known!'
                                         .format(file_path))
                        parameter_hashes.append(hash_parameter)

                        experiment_name = parameter["experiment_name"]
                        self.assertNotIn(experiment_name, experiment_names, 'Duplicated experiment name: {}'
                                         .format(experiment_name))

                        derived_filename = '{}.json'.format(experiment_name)
                        msg = 'Configuration {} does not match experiment name {}!'.format(filename, derived_filename)
                        self.assertEqual(filename, derived_filename, msg=msg)

                        experiment_names.append(experiment_name)

                    elif experiment_type == 'dict-based':
                        counter = counter + 1
                    elif experiment_type == 'eval-transformer-based':
                        counter = counter + 1

                        experiment_name = experiments["experiment_name"]
                        self.assertNotIn(experiment_name, experiment_names, 'Duplicated experiment name: {}'
                                         .format(experiment_name))
                        experiment_names.append(experiment_name)

                        derived_filename = '{}.json'.format(experiment_name)
                        msg = 'Configuration {} does not match experiment name {}!'.format(filename, derived_filename)
                        self.assertEqual(filename, derived_filename, msg=msg)

                        # Evaluate if original experiment exists
                        original_experiment_name = experiments["original_experiment_name"]
                        original_dataset = experiments["original_dataset"]

                        experiment_reference = (original_dataset, original_experiment_name, experiment_name)
                        original_experiment_dataset_names.append(experiment_reference)

                        prediction_output = experiments["prediction_output"]
                        self.assertNotIn(prediction_output, prediction_outputs, 'Duplicated prediction output: {}'
                                         .format(prediction_output))
                        prediction_outputs.append(prediction_output)

                dataset_experiment_name_combination[dataset_name] = experiment_names.copy()

            for original_dataset, original_experiment_name, experiment_name in original_experiment_dataset_names:
                self.assertIn(original_experiment_name, dataset_experiment_name_combination[original_dataset],
                              'Original experiment of evaluation experiment {} does not exist!'. format(experiment_name))

            self.assertEqual(n_configurations, counter,
                             'Some configurations of dataset {} have an unknown experiment type!'.format(dataset_name))


if __name__ == '__main__':
    unittest.main()
