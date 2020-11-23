import time
from datetime import datetime

from src.data.preprocessing import preprocess
from src.evaluation import scorer
from src.experiments.runner.experiment_runner import ExperimentRunner
from src.models.transformers import utils
from src.models.transformers.dataset.category_dataset_flat import CategoryDatasetFlat
from src.utils.result_collector import ResultCollector

from transformers import TrainingArguments, Trainer


class ExperimentRunnerTransformer(ExperimentRunner):

    def __init__(self, path, test, experiment_type):
        super().__init__(path, test, experiment_type)

        self.load_experiments(path)
        self.load_datasets()

        self.load_tree()

    def load_experiments(self, path):
        """Load experiments defined in the json for which a path is provided"""
        experiments = self.load_configuration(path)
        self.parameter = experiments['parameter']

    def encode_labels(self):
        """Encode & decode labels plus rescale encoded values"""
        normalized_encoder = {}
        normalized_decoder = {}
        decoder = dict(self.tree.nodes(data="name"))
        encoder = dict([(value, key) for key, value in decoder.items()])

        leaf_nodes = [node[0] for node in self.tree.out_degree(self.tree.nodes()) if node[1] == 0]
        leaf_nodes = [decoder[node] for node in leaf_nodes]
        number_leaf_nodes = len(leaf_nodes)

        # Rescale keys!
        counter = 0
        for key in encoder:
            if key in leaf_nodes:
                normalized_encoder[key] = {'original_key': encoder[key], 'derived_key': counter}
                normalized_decoder[counter] = {'original_key': encoder[key], 'value': key}
                counter += 1

        return normalized_encoder, normalized_decoder, number_leaf_nodes

    def run(self):
        """Run experiments"""
        result_collector = ResultCollector(self.dataset_name, self.experiment_type)

        normalized_encoder, normalized_decoder, number_leaf_nodes = self.encode_labels()

        tokenizer, model = utils.provide_model_and_tokenizer(self.parameter['model_name'], number_leaf_nodes)

        tf_ds = {}
        for key in self.dataset:
            df_ds = self.dataset[key]
            if self.test:
                # load only subset of the data
                df_ds = df_ds[:50]
                self.logger.warning('Run in test mode - dataset reduced to 50 records!')

            if self.parameter['preprocessing'] == "True":
                texts = [preprocess(value) for value in df_ds['title'].values]
            else:
                texts = list(df_ds['title'].values)

            # Normalize label values
            labels = [value.replace(' ', '_') for value in df_ds['category'].values]

            tf_ds[key] = CategoryDatasetFlat(texts, labels, tokenizer, normalized_encoder)

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
                                              transformer_decoder=normalized_decoder)
        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=tf_ds['train'],  # tensorflow_datasets training dataset
            eval_dataset=tf_ds['validate'],  # tensorflow_datasets evaluation dataset
            compute_metrics=evaluator.compute_metrics_transformers_flat
        )

        trainer.train()

        for split in ['train', 'validate', 'test']:
            result_collector.results['{}+{}'.format(self.parameter['experiment_name'], split)] \
                = trainer.evaluate(tf_ds[split])

        trainer.save_model()

        # Persist results
        result_collector.persist_results(timestamp)
