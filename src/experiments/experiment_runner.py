import json
import logging
import sys

from src.models.dictionary.dictclassifier import DictClassifier


class ExperimentRunner:

    def __init__(self, path):
        self.logger = logging.getLogger(__name__)

        self.path = path
        self.experiment_type = None
        self.dataset = None
        self.parameter = None
        self.most_frequent_leaf = None

        self.results = None

        self.load(path)

    def __str__(self):
        output = 'Experiment runner for {} experiments on {} dataset with the following parameter: {}'\
            .format(self.experiment_type, self.dataset, self.parameter)
        return output

    def load(self, path):
        """Load experiments defined in the json for which a path is provided"""
        with open(path) as json_file:
            experiments = json.load(json_file)
            self.logger.info('Loaded experiments from {}'.format(path))

        self.dataset = experiments['dataset']
        self.experiment_type = experiments['type']
        if self.experiment_type == 'dict-based':
            self.most_frequent_leaf = experiments['most_frequent_leaf']

        # Normalise experiment parameter
        for parameters in experiments['parameter']:
            for parameter, value in parameters.items():
                if value == 'True':
                    parameters[parameter] = True
                elif value == 'False':
                    parameters[parameter] = False
        self.parameter = experiments['parameter']

    def run(self):
        """Run experiments"""
        if self.experiment_type == 'dict-based':
            dictclassifier = DictClassifier(self.dataset, self.most_frequent_leaf)

            # To-Do: Continue here! Implement experiment execution!!


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    if len(sys.argv) > 1:
        runner = ExperimentRunner(sys.argv[1])
        runner.run()
