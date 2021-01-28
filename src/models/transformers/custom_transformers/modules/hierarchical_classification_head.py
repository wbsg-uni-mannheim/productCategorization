import torch
from torch import nn


class HierarchicalClassificationHead(nn.Module):
    """Head for hierarchical classification tasks"""

    def __init__(self, config):
        super(HierarchicalClassificationHead, self).__init__()

        self.hidden_size = config.hidden_size
        self.paths = config.paths
        self.num_nodes = config.num_nodes

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # create a weight matrix and bias vector for each node in the tree
        self.nodes = nn.ModuleList([nn.Linear(self.hidden_size, 1) for i in range(self.num_nodes)])


    def forward(self, input):
        # Make a prediction along all paths in the tree
        logit_list =  [self.predict_along_path(input,path) for path in self.paths]

        logits = torch.stack(logit_list, dim=1)

        return logits

    def predict_along_path(self, input, path):
        input = self.dropout(input)

        # Make predictions along path
        logits = [torch.sigmoid(self.nodes[node](input)) for node in path]
        logits = torch.cat(logits, dim=1)

        # Calculate logit for given input and path
        logit = torch.prod(logits, dim=1)

        return logit