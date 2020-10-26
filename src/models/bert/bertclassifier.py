import logging
from transformers import TFBertForSequenceClassification, TFTrainer, TFTrainingArguments


class BERTClassifier(object):

    def __init__(self, dataset):
        self.logger = logging.getLogger(__name__)

        self.dataset = dataset
        self.model = TFBertForSequenceClassification.from_pretrained("bert-large-uncased")

        self.training_args = TFTrainingArguments(
            output_dir='./results',  # output directory
            num_train_epochs=3,  # total # of training epochs
            per_device_train_batch_size=16,  # batch size per device during training
            per_device_eval_batch_size=64,  # batch size for evaluation
            warmup_steps=500,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            logging_dir='./logs',  # directory for storing logs
        )

        self.trainer = TFTrainer(
            model=self.model,  # the instantiated 🤗 Transformers model to be trained
            args=self.training_args,  # training arguments, defined above
            train_dataset= train_dataset,  # training dataset
            eval_dataset= test_dataset  # evaluation dataset
        )
