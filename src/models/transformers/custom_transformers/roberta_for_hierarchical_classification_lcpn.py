from typing import Optional, Tuple

import networkx as nx
import torch
from dataclasses import dataclass
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaModel
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_roberta import RobertaPreTrainedModel

class RobertaHierarchyHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, num_labels):
        super(RobertaHierarchyHead, self).__init__()

        self.hidden_size = config.hidden_size
        self.num_labels = num_labels

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.i2h = nn.Linear(config.hidden_size + config.hidden_size, config.hidden_size)
        self.i2o = nn.Linear(config.hidden_size + config.hidden_size, self.num_labels)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        combined = self.dropout(combined)
        combined = torch.tanh(combined)

        hidden = self.i2h(combined)

        output = self.i2o(combined)

        return output, hidden


class RobertaForHierarchicalClassificationLCPN(RobertaPreTrainedModel):
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.hidden_size = config.hidden_size

        self.classifier_tree = nx.DiGraph()
        self.initialize_classifier_tree(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0][:, 0, :].view(-1,768)  # take <s> token (equiv. to [CLS])

        loss = None
        logits_list = []
        transposed_labels = torch.transpose(labels,0, 1)
        hidden = self.initHidden(len(labels), self.hidden_size)
        root = [node[0] for node in self.classifier_tree.in_degree if node[1] == 0][0]
        #Initialize RNNHead
        for node in self.classifier_tree.nodes(data='classifier'):
            id, classifier = node
            classifier.zero_grad()

        ready_for_training = [{'node': root, 'labels': transposed_labels[1], 'lvl': 0, 'hidden': hidden}]

        for i in range(len(transposed_labels)):

            ready_for_training_lvl = [training for training in ready_for_training if training['lvl'] == i]

            while len(ready_for_training_lvl) > 0:
                # Training
                training = ready_for_training_lvl.pop(0)
                classifier = [classifier for id, classifier in self.classifier_tree.nodes(data='classifier') if id == training['node']]
                logits, hidden = classifier(sequence_output, training['hidden'])
                num_labels = classifier.num_labels

                # Compute loss
                loss_fct = CrossEntropyLoss()
                if loss is None:
                    loss = loss_fct(logits.view(-1, num_labels), training['labels'].view(-1))
                else:
                    loss += loss_fct(logits.view(-1, num_labels), training['labels'].view(-1))

                # Prepare prediction

            logits_list.append(logits_lvl)

        logits = torch.stack(logits_list)
        logits = torch.transpose(logits, 0, 1)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    def initialize_classifier_tree(self, config):
        for id in config.tree.nodes():
            successors = [node for node in config.tree.successors(id)]
            num_labels = len(successors) + 1 # Each successor has one label as well as out of category
            if len(successors) > 1:
                classifier = RobertaHierarchyHead(config, num_labels)
                self.classifier_tree.add_node(id, classifier=classifier)

        for edge in config.tree.edges():
            parent, child = edge
            if self.classifier_tree.has_node(parent) and self.classifier_tree.has_node(child):
                self.classifier_tree.add_edge(parent, child)

    def initHidden(self, size, hidden_size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.zeros(size, hidden_size).to(device)