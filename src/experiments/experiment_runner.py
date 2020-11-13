import json
import logging
from pathlib import Path

import pandas as pd

from src.models.model_runner import ModelRunner


class ExperimentRunner(ModelRunner):

    def __init__(self, path, test, experiment_type):

        super().__init__(path, test, experiment_type)

    def __str__(self):
        output = 'Experiment runner for {} experiments on {} dataset with the following parameter: {}' \
            .format(self.experiment_type, self.dataset_name, self.parameter)
        return output

    def run(self):
        """Run experiments - Implemented in child classes!"""
