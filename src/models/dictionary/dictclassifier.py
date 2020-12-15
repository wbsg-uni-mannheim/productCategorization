"""Dictionary based approach --> serves as baseline"""
import logging
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


class DictClassifier(object):

    def __init__(self, dataset, most_frequent_leaf, tree):

        self.logger = logging.getLogger(__name__)

        self.dataset = dataset
        self.synonyms_dict = None
        self.most_frequent_leaf = most_frequent_leaf
        self.tree = tree

        self.wnl = WordNetLemmatizer()
        self.generate_synonyms(dataset)
        self.logger.info('Initialized Dict-classifier for dataset {}'.format(dataset))



    def generate_synonyms(self, dataset):

        nltk.download('wordnet')

        leaf_nodes_wdc = [node[0] for node in self.tree.out_degree(self.tree.nodes()) if node[1] == 0]
        decoder = dict(self.tree.nodes(data="name"))
        leaf_nodes_wdc = [decoder[node] for node in leaf_nodes_wdc]

        self.synonyms_dict = {}

        for classname in leaf_nodes_wdc:
            synonyms = []

            for syn in wordnet.synsets(classname.replace(' ', '_')):
                for lem in syn.lemmas():
                    synonyms.append(lem.name().replace('_', ' ').lower())
            self.synonyms_dict[classname] = synonyms

        self.logger.info('Loaded synonyms for dataset {}'.format(dataset))

    def count_occurrences(self, list_of_words, target_string):
        if len(list_of_words) == 0:
            return 0
        else:
            # division by len(list_of_words) because if a label has multiple words, its count is likely to be higher
            count = sum([target_string.count(word) for word in list_of_words])
            return count / len(list_of_words)

    def classify_dictionary_based(self, test_data, fallback_classifier, synonyms, lemmatize):
        y_pred = []
        count = 0
        # Prepare class strings
        classes_prep = {}
        for classname in self.synonyms_dict.keys():
            classes_prep[classname] = set(classname.lower().split())
            if synonyms:
                classes_prep[classname] = classes_prep[classname].union(
                    set(' '.join(self.synonyms_dict[classname]).split()))
            if lemmatize:
                classes_prep[classname] = [self.wnl.lemmatize(word) for word in classes_prep[classname]]

        # Prepare class strings
        to_predict_by_fallback = []
        for text_of_instance in test_data:
            max_word = None
            max_word_count = 0
            for classname in self.synonyms_dict.keys():
                word_rel_count = self.count_occurrences(classes_prep[classname], text_of_instance)
                if word_rel_count > max_word_count:
                    max_word_count = word_rel_count
                    max_word = classname

            if max_word_count == 0:
                count = count + 1
                if fallback_classifier:
                    to_predict_by_fallback.append(text_of_instance)
                    y_pred.append(0)
                else:
                    y_pred.append(self.most_frequent_leaf)
            else:
                y_pred.append(max_word)

        if fallback_classifier and len(to_predict_by_fallback) > 0:
            # Use fallback only in case it is necessary!
            fallback_predictions = fallback_classifier.predict(to_predict_by_fallback)

            for i, p in enumerate(y_pred):
                if p == 0:
                    y_pred[i] = fallback_predictions[0]
                    fallback_predictions = fallback_predictions[1:]

        self.logger.info('Most frequent class/fallback classifier was used %d times' % count)

        return y_pred
