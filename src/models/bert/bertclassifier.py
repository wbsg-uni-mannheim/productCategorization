import logging


class BERTClassifier(object):

    def __init__(self, dataset):

        self.logger = logging.getLogger(__name__)

        self.dataset = dataset
