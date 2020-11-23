from typing import Optional, Tuple

import torch
from dataclasses import dataclass
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaModel
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_roberta import RobertaPreTrainedModel

class RobertaRNNHead(nn.Module):
    """Head for sentence-level classification tasks."""

    #def __init__(self, config):
    #    super().__init__()
    #    self.dense = nn.Linear(config.hidden_size, config.hidden_size)
    #    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    #    self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def __init__(self, config):
        super(RobertaRNNHead, self).__init__()

        self.hidden_size = config.hidden_size

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.i2h = nn.Linear(config.hidden_size + config.hidden_size, config.hidden_size)
        self.i2o = nn.Linear(config.hidden_size + config.hidden_size, config.num_labels)
        self.softmax = nn.LogSoftmax(dim=1)

    #def forward(self, features, hidden, **kwargs):
    #    x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
    #    x = self.dropout(x)
    #    x = self.dense(x)
    #    x = torch.tanh(x)
    #    x = self.dropout(x)
    #    x = self.out_proj(x)
    #    return x


    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        combined = self.dropout(combined)
        combined = torch.tanh(combined)

        hidden = self.i2h(combined)

        output = self.i2o(combined)

        return output, hidden

    def initHidden(self, size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.zeros(size, self.hidden_size).to(device)

class RobertaForHierarchicalClassificationRNN(RobertaPreTrainedModel):
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)


        self.classifier = RobertaRNNHead(config)

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
        hidden = self.classifier.initHidden(len(labels))
        #Initialize RNNHead
        self.classifier.zero_grad()

        for i in range(len(transposed_labels)):
            logits_lvl, hidden = self.classifier(sequence_output, hidden)

            logits_list.append(logits_lvl)

            loss_fct = CrossEntropyLoss()
            if loss is None:
                loss = loss_fct(logits_list[i].view(-1, self.num_labels), transposed_labels[i].view(-1))
            else:
                loss += loss_fct(logits_list[i].view(-1, self.num_labels), transposed_labels[i].view(-1))

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

