from torch import nn
import torch.nn.functional as F
import torch


class RNN(nn.Module):
    # Pytorch prototype example
    def __init__(self, input_dim, hidden_dim, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_dim
        self.combined_input_size = hidden_dim + input_dim

        self.input_to_hidden = nn.Linear()

        self.rnn_cell = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True,
                          nonlinearity='relu')

        self.output = output_size

    def forward(self, data, last_hidden):

        last_hidden = data + last_hidden
        # One time step
        out, hidden_current = self.rnn_cell(data, last_hidden)




        return out, hidden_current

    def train(self, x, y, batch_size, loss=nn.MSELoss, iterations=100):
        loss = 0

        batched_input = (batch_size, x.shape[1])
        hidden_size = (batch_size, self.hidden_size)
        target_size = (batch_size, y)

        for t in range(iterations):
            hidden, output = self(batch_size, )
