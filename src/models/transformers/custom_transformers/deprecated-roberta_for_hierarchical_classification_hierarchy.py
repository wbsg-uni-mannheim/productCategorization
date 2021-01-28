import torch
from torch.nn import NLLLoss
from transformers import RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_roberta import RobertaPreTrainedModel

from src.models.transformers.custom_transformers.modules.focal_loss import FocalLoss
from src.models.transformers.custom_transformers.modules.roberta_rnn_head import RobertaRNNHead


class RobertaForHierarchicalClassificationHierarchy(RobertaPreTrainedModel):
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels_per_level = config.num_labels_per_level

        self.roberta = RobertaModel(config, add_pooling_layer=False)

        self.classifier_dict = {}
        self.max_no_labels_per_lvl = 0

        self.next_labels_on_level = config.next_labels_on_level

        self.hierarchy_certainty = config.hierarchy_certainty
        self.focal_loss = config.focal_loss

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for level in config.num_labels_per_level:
            num_labels = config.num_labels_per_level[level]
            if num_labels > self.max_no_labels_per_lvl:
                self.max_no_labels_per_lvl = num_labels
            self.classifier_dict[level] = RobertaRNNHead(config, num_labels).to(device)

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

        sequence_output = outputs[0][:, 0, :].view(-1, 768)  # take <s> token (equiv. to [CLS])

        loss = None
        logits_list = []
        transposed_labels = torch.transpose(labels, 0, 1)
        batch_size = len(labels)
        hidden = self.classifier_dict[1].initHidden(batch_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize RNNHead
        for classifier in self.classifier_dict.values():
            classifier.zero_grad()

        for i in range(len(transposed_labels)):
            logits_lvl, hidden = self.classifier_dict[i + 1](sequence_output, hidden)

            # Masking is not used for now!! --> uncomment if needed again
            # if i > 0:
            #     successors_per_node = self.next_labels_on_level[i]
            #    mask = []
            #     for prob, label in zip(previous_prediction_prob, previous_predictions_label):
            #         if prob.item() > math.log(self.hierarchy_certainty):
            #             # Do masking only if the previous prediction was sure!
            #             mask_values = [0] * self.num_labels_per_level[i + 1]
            #             mask_values[0] = 1
            #             prediction_value = label.item()
            #             if prediction_value != 0:
            #                 successors = successors_per_node[prediction_value]
            #                 for successor in successors:
            #                     mask_values[successor] = 1
            #         else:
            #             mask_values = [1] * self.num_labels_per_level[i + 1]
            #
            #         mask_values = torch.FloatTensor(mask_values).to(device)
            #         mask.append(mask_values)
            #
            #     mask_tensor = torch.stack(mask).to(device)
            #
            #     logits_lvl = self.masked_vector(logits_lvl, mask_tensor)
            #
            # previous_prediction_prob, previous_predictions_label = torch.max(logits_lvl, 1)

            if self.focal_loss:
                loss_fct = FocalLoss()
            else:
                loss_fct = NLLLoss()

            if loss is None:
                loss = loss_fct(logits_lvl.view(-1, self.num_labels_per_level[i + 1]), transposed_labels[i].view(-1))
            else:
                loss += loss_fct(logits_lvl.view(-1, self.num_labels_per_level[i + 1]), transposed_labels[i].view(-1))

            num_added_zeros = self.max_no_labels_per_lvl - self.num_labels_per_level[i + 1]
            all_zeros = torch.zeros(batch_size, num_added_zeros).to(device)
            logits_lvl = torch.cat((logits_lvl, all_zeros), dim=1)

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

#    ---> Unused vector masking <---
#    def masked_vector(self, vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
#        """
#        https://github.com/allenai/allennlp/blob/b6cc9d39651273e8ec2a7e334908ffa9de5c2026/allennlp/nn/util.py#L272-L303
#        ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
#        masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
#        ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.
#        ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
#        broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
#        unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
#        do it yourself before passing the mask into this function.
#        In the case that the input vector is completely masked, the return value of this function is
#        arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
#        of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
#        that we deal with this case relies on having single-precision floats; mixing half-precision
#        floats with fully-masked vectors will likely give you ``nans``.
#        If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
#        lower), the way we handle masking here could mess you up.  But if you've got logit values that
#        extreme, you've got bigger problems than this.
#        """
#        if mask is not None:
#            mask = mask.float()
#            while mask.dim() < vector.dim():
#                mask = mask.unsqueeze(1)
#            # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
#            # results in nans when the whole vector is masked.  We need a very small value instead of a
#            # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
#            # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
#            # becomes 0 - this is just the smallest value we can actually use.
#            vector = vector + (mask + 1e-45).log()
#        return vector
