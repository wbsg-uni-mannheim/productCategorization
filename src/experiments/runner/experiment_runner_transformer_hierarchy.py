import time
from datetime import datetime

from src.data.preprocessing import preprocess
from src.evaluation import scorer
from src.experiments.runner.experiment_runner import ExperimentRunner
from src.models.transformers import utils
from src.models.transformers.dataset.category_dataset_multi_label import CategoryDatasetMultiLabel
from src.utils.result_collector import ResultCollector

from transformers import TrainingArguments, Trainer, RobertaConfig


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

    def determine_path_to_root(self, nodes):
        predecessors = self.tree.predecessors(nodes[-1])
        predecessor = [k for k in predecessors][0]

        if predecessor == self.root:
            nodes.reverse()
            return nodes
        nodes.append(predecessor)
        return self.determine_path_to_root(nodes)

    def normalize_path_from_root_per_level(self, path):
        """Normalize label values per level"""
        normalized_path = []
        for i in range(len(path)):
            counter = 0
            nodes_per_lvl = self.get_all_nodes_per_lvl(i)
            for node in nodes_per_lvl:
                counter += 1
                if node == path[i]:
                    normalized_path.append(counter)
                    break

        assert (len(path) == len(normalized_path))
        return normalized_path

    def get_all_nodes_per_lvl(self, level):
        successors = self.tree.successors(self.root)
        while level > 0:
            next_lvl_succesors = []
            for successor in successors:
                next_lvl_succesors.extend(self.tree.successors(successor))
            successors = next_lvl_succesors
            level -= 1
        return successors

    def encode_labels(self):
        """Encode & decode labels plus rescale encoded values"""
        normalized_encoder = {}
        normalized_decoder = {}
        decoder = dict(self.tree.nodes(data="name"))
        encoder = dict([(value, key) for key, value in decoder.items()])

        leaf_nodes = [node[0] for node in self.tree.out_degree(self.tree.nodes()) if node[1] == 0]
        leaf_nodes = [decoder[node] for node in leaf_nodes]

        counter = 0
        longest_path = 0
        for key in encoder:
            if key in leaf_nodes:
                path = self.determine_path_to_root([encoder[key]])

                #Normalize Path per level in hierarchy
                path = self.normalize_path_from_root_per_level(path)

                normalized_encoder[key] = {'original_key': encoder[key], 'derived_key': counter,
                                           'derived_path': path}
                normalized_decoder[counter] = {'original_key': encoder[key], 'value': key}
                counter += 1
                if len(path) > longest_path:
                    longest_path = len(path)

        # Number of labels per level is determined via the tree
        num_labels_per_level = {}
        next_labels_on_level = {}
        for i in range(longest_path):
            # Number of labels per level plus 1 for out of hierarchy nodes
            node_level = [node for node in self.get_all_nodes_per_lvl(i)]
            num_labels_per_level[i+1] = len(node_level) + 1

            # Determine encoded successors
            node_level_plus_one = [node for node in self.get_all_nodes_per_lvl(i+1)]
            if len(node_level_plus_one) > 0:
                encoded_successors_per_node = {}
                for j in range(len(node_level)):
                    node = node_level[j]
                    successors = self.tree.successors(node)
                    encoded_successors = []
                    for succesor in successors:
                        if succesor in node_level_plus_one:
                            encoded_successor = node_level_plus_one.index(succesor) + 1 # 0 is out of hierarchy
                            encoded_successors.append(encoded_successor)

                    encoded_successors_per_node[j+1] = encoded_successors
                next_labels_on_level[i+1] = encoded_successors_per_node

        return normalized_encoder, normalized_decoder, num_labels_per_level, next_labels_on_level

    def run(self):
        """Run experiments"""
        result_collector = ResultCollector(self.dataset_name, self.experiment_type)

        normalized_encoder, normalized_decoder, num_labels_per_level, next_labels_on_level = self.encode_labels()

        config = RobertaConfig.from_pretrained("roberta-base")
        config.num_labels_per_level = num_labels_per_level
        config.next_labels_on_level = next_labels_on_level

        tokenizer, model = utils.provide_model_and_tokenizer(self.parameter['model_name'], config=config)

        tf_ds = {}
        for key in self.dataset:
            df_ds = self.dataset[key]
            if self.test:
                # load only subset of the data
                df_ds = df_ds[:20]
                self.logger.warning('Run in test mode - dataset reduced to 20 records!')

            if self.parameter['preprocessing'] == "True":
                texts = [preprocess(value) for value in df_ds['title'].values]
            else:
                texts = list(df_ds['title'].values)

            # Normalize label values
            labels = [value.replace(' ', '_') for value in df_ds['category'].values]

            tf_ds[key] = CategoryDatasetMultiLabel(texts, labels, tokenizer, normalized_encoder)

        timestamp = time.time()
        string_timestamp = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d_%H-%M-%S')
        training_args = TrainingArguments(
            output_dir='./experiments/{}/transformers/model/{}'
                .format(self.dataset_name, self.parameter['experiment_name']),
            # output directory
            num_train_epochs=self.parameter['epochs'],  # total # of training epochs
            learning_rate=self.parameter['learning_rate'],
            per_device_train_batch_size=self.parameter['per_device_train_batch_size'],
            # batch size per device during training
            per_device_eval_batch_size=64,  # batch size for evaluation
            warmup_steps=500,  # number of warmup steps for learning rate scheduler
            weight_decay=self.parameter['weight_decay'],  # strength of weight decay
            logging_dir='./experiments/{}/transformers/logs-{}'.format(self.dataset_name, string_timestamp),
            # directory for storing logs
            save_total_limit=5,  # Save only the last 5 Checkpoints
            metric_for_best_model=self.parameter['metric_for_best_model'],
            load_best_model_at_end=True,
            gradient_accumulation_steps=self.parameter['gradient_accumulation_steps'],
            seed=self.parameter['seed']
        )

        evaluator = scorer.HierarchicalScorer(self.parameter['experiment_name'], self.tree,
                                              transformer_decoder=normalized_decoder, num_labels_per_level= num_labels_per_level)
        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=tf_ds['train'],  # tensorflow_datasets training dataset
            eval_dataset=tf_ds['validate'],  # tensorflow_datasets evaluation dataset
            compute_metrics=evaluator.compute_metrics_transformers_hierarchy
        )

        trainer.train()

        for split in ['train', 'validate', 'test']:
            result_collector.results['{}+{}'.format(self.parameter['experiment_name'], split)] \
                = trainer.evaluate(tf_ds[split])

        trainer.save_model()

        # Persist results
        result_collector.persist_results(timestamp)
