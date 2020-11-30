import logging
from pathlib import Path
import os
from datetime import datetime
import csv


class ResultCollector():

    def __init__(self, dataset_name, experiment_type):
        self.logger = logging.getLogger(__name__)

        self.results = {}
        self.dataset_name = dataset_name
        self.experiment_type = experiment_type

    def persist_results(self, timestamp):
        """Persist Experiment Results"""
        project_dir = Path(__file__).resolve().parents[2]
        relative_path = 'results/{}/'.format(self.dataset_name)
        absolute_path = project_dir.joinpath(relative_path)

        if not os.path.exists(absolute_path):
            os.mkdir(absolute_path)

        file_path = absolute_path.joinpath('{}_{}_results_{}.csv'.format(
                            self.dataset_name, self.experiment_type,
                            datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d_%H-%M-%S')))

        header = ['Experiment Name','Dataset','Split']
        # Use first experiment as reference for the metric header
        metric_header = list(list(self.results.values())[0].keys())
        header = header + metric_header

        rows = []
        for result in self.results.keys():
            if '+' in result:
                result_parts = result.split('+')
                experiment_name = result_parts[0]
                split = result_parts[1]
                row = [experiment_name, self.dataset_name, split]
            else:
                row = [result, self.dataset_name, 'all']
            for metric in self.results[result].items():
                row.append(metric[1])
            rows.append(row)

        # Write to csv
        with open(file_path, 'w', newline='') as f:
            csv_writer = csv.writer(f, delimiter=';')

            csv_writer.writerow(header)
            csv_writer.writerows(rows)

        self.logger.info('Results of {} on {} written to file {}!'.format(
                            self.experiment_type, self.dataset_name, file_path.absolute()))