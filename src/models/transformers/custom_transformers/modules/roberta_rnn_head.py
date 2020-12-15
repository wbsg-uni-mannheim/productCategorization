import torch
from torch import nn


class RobertaRNNHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, num_labels):
        super(RobertaRNNHead, self).__init__()

        self.hidden_size = config.hidden_size

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.i2h = nn.Linear(config.hidden_size + config.hidden_size, config.hidden_size)
        self.i2o = nn.Linear(config.hidden_size + config.hidden_size, num_labels)

        self.o2o = nn.Linear(config.hidden_size + config.num_labels, config.num_labels)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        combined = self.dropout(combined)

        hidden = self.i2h(combined)
        hidden = torch.tanh(hidden)

        output = self.i2o(combined)
        output = torch.tanh(output)

        output_combined = torch.cat((hidden, output), 1)
        output = self.dropout(output_combined)
        output = self.o2o(output)

        return output, hidden

    def initHidden(self, size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.zeros(size, self.hidden_size).to(device)