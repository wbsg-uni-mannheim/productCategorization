"""Dictionary based approach --> serves as baseline"""
import logging
import pickle
from pathlib import Path

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


class DictClassifier(object):

    def __init__(self, dataset, most_frequent_leaf):

        self.logger = logging.getLogger(__name__)

        self.dataset = dataset
        self.synonyms_dict = self.generate_synonyms(dataset)
        self.most_frequent_leaf = most_frequent_leaf

        self.wnl = WordNetLemmatizer()
        self.logger.info('Initialized Dict-classifier for dataset {}'.format(dataset))

    def generate_synonyms(self, dataset):
        project_dir = Path(__file__).resolve().parents[3]
        path_to_tree = project_dir.joinpath('data', 'raw', dataset, 'tree', 'tree_{}.pkl'.format(dataset))

        with open(path_to_tree, 'rb') as f:
            tree = pickle.load(f)

        leaf_nodes_wdc = [node[0] for node in tree.out_degree if node[1] == 0]

        synonyms_dict = {}

        for classname in leaf_nodes_wdc:
            synonyms = []

            for syn in wordnet.synsets(classname.replace(' ', '_')):
                for lem in syn.lemmas():
                    synonyms.append(lem.name().replace('_', ' ').lower())
            synonyms_dict[classname] = synonyms

        self.logger.info('Loaded synonyms for dataset {}'.format(dataset))
        return synonyms_dict

    def count_occurrences(self, list_of_words, target_string, lemmatize=False):
        count = 0
        for word in list_of_words:
            if lemmatize:
                word = self.wnl.lemmatize(word)
            count = count + target_string.count(word)

        # division by len(list_of_words) because if a label has multiple words, its count is likely to be higher
        if len(list_of_words) > 0:
            return count / len(list_of_words)
        else:
            return 0

    def classify_dictionary_based(self, test_data, fallback_classifier, synonyms=False, lemmatize=False):
        y_pred = []
        count = 0

        for text_of_instance in test_data:
            word_count = {}

            for classname in self.synonyms_dict.keys():
                words_in_classname = set(classname.lower().split())
                if synonyms:
                    words_in_classname = words_in_classname.union(set(' '.join(self.synonyms_dict[classname]).split()))
                word_count[classname] = self.count_occurrences(words_in_classname, text_of_instance, lemmatize)

                max_words = max(word_count, key=word_count.get)

                if word_count[max_words] == 0:
                    count = count + 1
                    if fallback_classifier:
                        y_pred.append(fallback_classifier.predict(text_of_instance))
                    else:
                        y_pred.append(self.most_frequent_leaf)
                else:
                    y_pred.append(max(word_count, key=word_count.get))

        self.logger.info('Most frequent class/fallback classifier was used %d times' % count)

        return y_pred
