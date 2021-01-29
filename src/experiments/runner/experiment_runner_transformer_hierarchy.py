import time
from datetime import datetime

from src.data.preprocessing import preprocess
from src.evaluation import scorer
from src.experiments.runner.experiment_runner import ExperimentRunner
from src.models.transformers import utils
from src.models.transformers.dataset.category_dataset_flat import CategoryDatasetFlat
from src.utils import tree_utils
from src.utils.result_collector import ResultCollector

from transformers import TrainingArguments, Trainer, RobertaConfig

from src.utils.tree_utils import TreeUtils


class ExperimentRunnerTransformerHierarchy(ExperimentRunner):

    def __init__(self, path, test, experiment_type):
        super().__init__(path, test, experiment_type)

        self.load_experiments(path)
        self.load_datasets()

        self.load_tree()

    def load_experiments(self, path):
        """Load experiments defined in the json for which a path is provided"""
        experiments = self.load_configuration(path)
        self.parameter = experiments['parameter']

    # def encode_labels(self):
    #     """Encode & decode labels plus rescale encoded values"""
    #     # Use root node for out of vocabulary prediction
    #     normalized_encoder = {'Root': {'original_key': 0, 'derived_key': 0}}
    #     normalized_decoder = { 0: {'original_key': 0, 'value': 'Root'}}
    #     decoder = dict(self.tree.nodes(data="name"))
    #     encoder = dict([(value, key) for key, value in decoder.items()])
    #
    #     leaf_nodes = [node[0] for node in self.tree.out_degree(self.tree.nodes()) if node[1] == 0]
    #     leaf_nodes = [decoder[node] for node in leaf_nodes]
    #
    #     new_leaf_nodes = self.tree_utils.get_sorted_leaf_nodes()
    #
    #     # Rescale keys!
    #     derived_key = 1 # Start with 1 --> 0 is out of category
    #     for key in encoder:
    #         if key in leaf_nodes:
    #             normalized_encoder[key] = {'original_key': encoder[key], 'derived_key': derived_key}
    #             normalized_decoder[derived_key] = {'original_key': encoder[key], 'value': key}
    #             derived_key += 1
    #
    #     return normalized_encoder, normalized_decoder, number_leaf_nodes

    def intialize_hierarchy_paths(self):
        """initialize paths using the provided tree"""

        leaf_nodes = [node[0] for node in self.tree.out_degree if node[1] == 0]
        paths = [self.tree_utils.determine_path_to_root([node]) for node in leaf_nodes]

        # Normalize paths per level in hierarchy - currently the nodes are of increasing number throughout the tree.
        normalized_paths = [self.tree_utils.normalize_path_from_root_per_level(path) for path in paths]


        normalized_encoder = {'Root': {'original_key': 0, 'derived_key': 0}}
        normalized_decoder = { 0: {'original_key': 0, 'value': 'Root'}}
        decoder = dict(self.tree.nodes(data="name"))
        encoder = dict([(value, key) for key, value in decoder.items()])

        #initiaize encoders
        for path, normalized_path in zip(paths, normalized_paths):
            key = path[-1]
            derived_key = normalized_path[-1]
            if key in leaf_nodes:
                normalized_encoder[decoder[key]] = {'original_key': key, 'derived_key': derived_key}
                normalized_decoder[derived_key] = {'original_key': key, 'value': decoder[key]}

        oov_path = [[0, 0, 0]]
        normalized_paths = oov_path + normalized_paths

        # Sort paths by last node in list
        sorted_normalized_paths = []
        for i in range(len(normalized_paths)):
            found_path = None
            for path in normalized_paths:
                if path[-1] == i:
                    found_path = path
                    break

            if not (found_path is None):
                sorted_normalized_paths.append(found_path)
                normalized_paths.remove(found_path)

        return normalized_encoder, normalized_decoder, sorted_normalized_paths

    def run(self):
        """Run experiments"""
        result_collector = ResultCollector(self.dataset_name, self.experiment_type)

        normalized_encoder, normalized_decoder, sorted_normalized_paths = self.intialize_hierarchy_paths()

        config = RobertaConfig.from_pretrained("roberta-base")
        config.paths = sorted_normalized_paths
        config.num_labels_per_lvl = self.tree_utils.get_number_of_nodes_lvl()

        tokenizer, model = utils.provide_model_and_tokenizer(self.parameter['model_name'],
                                                             self.parameter['pretrained_model_or_path'],
                                                             config)

        tf_ds = {}
        for key in self.dataset:
            df_ds = self.dataset[key]
            if self.test:
                # load only subset of the data

                #DONT COMMIT CHANGE
                df_ds = df_ds[df_ds['category'] == '67010100_Clothing Accessories']

                df_ds = df_ds[:20]
                self.logger.warning('Run in test mode - dataset reduced to 20 records!')

            if self.parameter['description']:
                texts = list((df_ds['title'] + ' - ' + df_ds['description']).values)
            else:
                texts = df_ds['title'].values

            if self.parameter['preprocessing'] == True:
                texts = [preprocess(value) for value in texts]


            # Normalize label values
            labels = [value.replace(' ', '_') for value in df_ds['category'].values]

            tf_ds[key] = CategoryDatasetFlat(texts, labels, tokenizer, normalized_encoder)

        timestamp = time.time()
        string_timestamp = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d_%H-%M-%S')
        training_args = TrainingArguments(
            output_dir='{}/models/{}/transformers/model/{}'
                .format(self.data_dir, self.dataset_name, self.parameter['experiment_name']),
            # output directory
            num_train_epochs=self.parameter['epochs'],  # total # of training epochs
            learning_rate=self.parameter['learning_rate'],
            per_device_train_batch_size=self.parameter['per_device_train_batch_size'],
            # batch size per device during training
            per_device_eval_batch_size=64,  # batch size for evaluation
            warmup_steps=500,  # number of warmup steps for learning rate scheduler
            weight_decay=self.parameter['weight_decay'],  # strength of weight decay
            logging_dir='{}/models/{}/transformers/logs-{}'.format(self.data_dir, self.dataset_name, string_timestamp),
            # directory for storing logs
            save_total_limit=5,  # Save only the last 5 Checkpoints
            metric_for_best_model=self.parameter['metric_for_best_model'],
            load_best_model_at_end=True,
            gradient_accumulation_steps=self.parameter['gradient_accumulation_steps'],
            seed=self.parameter['seed'],
            disable_tqdm=False
        )

        evaluator = scorer.HierarchicalScorer(self.parameter['experiment_name'], self.tree,
                                              transformer_decoder=normalized_decoder)
        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=tf_ds['train'],  # tensorflow_datasets training dataset
            eval_dataset=tf_ds['validate'],  # tensorflow_datasets evaluation dataset
            compute_metrics=evaluator.compute_metrics_transformers_flat
        )

        self.logger.info('Start training!')
        trainer.train()

        for split in ['train', 'validate', 'test']:
            result_collector.results['{}+{}'.format(self.parameter['experiment_name'], split)] \
                = trainer.evaluate(tf_ds[split])

        trainer.save_model()

        # Persist results
        result_collector.persist_results(timestamp)
