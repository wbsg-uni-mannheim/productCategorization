import logging
import sys

from src.experiments.experiment_runner import ExperimentRunner

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    if len(sys.argv) > 1:
        runner = ExperimentRunner(sys.argv[1])
        runner.run()
