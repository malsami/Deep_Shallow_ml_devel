from torch import nn
import torch.nn.functional as F
import torch

# Use GPU if available, otherwise use cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# True if CUDA is available else false
use_cuda = torch.cuda.is_available()


# Bidirectional neural network is a recurrent neural network forwards and backwards
class BiRNN(nn.module):
    def __init__(self,input_size, hidden_size, num_layers, num_classes ):
        super(BiRNN,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True,bidirectional=True)
        self.fc = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, x):
        pass



