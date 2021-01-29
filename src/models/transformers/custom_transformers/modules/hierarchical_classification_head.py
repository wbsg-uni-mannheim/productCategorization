import torch
from torch import nn
from torch.nn import CrossEntropyLoss


class HierarchicalClassificationHead(nn.Module):
    """Head for hierarchical classification tasks"""

    def __init__(self, config):
        super(HierarchicalClassificationHead, self).__init__()

        self.hidden_size = config.hidden_size
        self.paths_per_lvl = self.initialize_paths_per_lvl(config.paths)

        self.num_labels_per_lvl = config.num_labels_per_lvl

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # create a weight matrix and bias vector for each node in the tree
        self.nodes = {}
        for lvl in self.num_labels_per_lvl:
            self.nodes[lvl] = nn.ModuleList([nn.Linear(self.hidden_size, 1) for i in range(self.num_labels_per_lvl[lvl])])


    def forward(self, input, labels):
        # Make a prediction along all paths in the tree
        loss_fct = CrossEntropyLoss()
        loss = None
        logits = None
        for lvl in self.paths_per_lvl:
            logit_list = [self.predict_along_path(input,path, lvl) for path in self.paths_per_lvl[lvl]]
            logits = torch.stack(logit_list, dim=1).to(self.device)

            updated_labels = self.update_label_per_lvl(labels, lvl)

            if loss is None:
                loss = loss_fct(logits.view(-1, self.num_labels_per_lvl[lvl]), updated_labels.view(-1))
            else:
                loss += loss_fct(logits.view(-1, self.num_labels_per_lvl[lvl]), updated_labels.view(-1))

        #Return only logits of last run to receive only valid paths!
        return logits, loss

    def predict_along_path(self, input, path, lvl):
        input = self.dropout(input)

        # Make predictions along path
        logits = [torch.sigmoid(self.nodes[lvl][node](input)).to(self.device) for node in path]
        logits = torch.cat(logits, dim=1).to(self.device)

        # Calculate logit for given input and path
        logit = torch.prod(logits, dim=1)

        return logit

    def initialize_paths_per_lvl(self, paths):
        length = max([len(path) for path in paths])
        added_paths = set()
        paths_per_lvl = {}
        for i in range(length):
            paths_per_lvl[i+1] = []
            for path in paths:
                new_path = path[:i+1]
                new_tuple = tuple(new_path)
                if not (new_tuple in added_paths):
                    added_paths.add(new_tuple)
                    paths_per_lvl[i+1].append(new_path)

        return paths_per_lvl

    def update_label_per_lvl(self, labels, lvl):
        #Move this function out of training in the future!!!!
        unique_values = labels
        updated_labels = labels.clone()
        for value in unique_values:
            searched_path = self.paths_per_lvl[len(self.paths_per_lvl)][value]
            update_value = searched_path[lvl - 1]
            updated_labels[updated_labels==value] = update_value

        return updated_labels