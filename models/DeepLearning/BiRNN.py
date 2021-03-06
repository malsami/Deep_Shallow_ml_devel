from torch import nn
import torch.nn.functional as F
import torch

"""Taken and inspired from pytorch intermediate tutorial on BiRNN
    https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/bidirectional_recurrent_neural_network/main.py
"""

# Use GPU if available, otherwise use cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# True if CUDA is available else false
use_cuda = torch.cuda.is_available()


# Bidirectional neural network is a recurrent neural network forwards and backwards
class BiRNN(nn.module):
    def __init__(self,input_size, hidden_size = 128, num_layers=2, num_classes=1):
        super(BiRNN,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size,num_layers, batch_first=True,bidirectional=True)
        self.fc = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out




