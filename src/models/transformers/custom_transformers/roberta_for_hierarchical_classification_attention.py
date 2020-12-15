import random
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_roberta import RobertaPreTrainedModel

class RobertaAttRNNHead(nn.Module):
    """Inspired by https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html"""

    def __init__(self, config):
        super(RobertaAttRNNHead, self).__init__()

        self.hidden_size = config.hidden_size
        self.output_size = config.num_labels
        self.max_length = config.max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)


    def forward(self, input, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]

        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        repeated_hidden = hidden.unsqueeze(1).repeat(-1, src_len, -1)

        combined = torch.cat((embedded, hidden), 1)
        attn_weights = F.softmax(
            self.attn(combined), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self, batch_size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.zeros(1, batch_size, self.hidden_size).to(device)

class RobertaForHierarchicalClassificationAttRNN(RobertaPreTrainedModel):
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)

        self.classifier = RobertaAttRNNHead(config)

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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_length = len(transposed_labels)
        target_length = len(transposed_labels)

        hierarchy_input = torch.tensor([0]*len(labels), device=device)

        #Initialize AttentionRNNHead
        self.classifier.zero_grad()

        teacher_forcing_ratio = 0.5 # Set to fixed value for now
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        for i in range(len(transposed_labels)):
            logits_lvl, hidden, attention = self.classifier(hierarchy_input, hidden, outputs[0])

            logits_list.append(logits_lvl)

            loss_fct = CrossEntropyLoss()
            if loss is None:
                loss = loss_fct(logits_list[i].view(-1, self.num_labels), transposed_labels[i].view(-1))
            else:
                loss += loss_fct(logits_list[i].view(-1, self.num_labels), transposed_labels[i].view(-1))

            if use_teacher_forcing:
                hierarchy_input = transposed_labels[i]
            else:
                preds = logits_lvl.argmax(-1)
                hierarchy_input = preds.detach()

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

