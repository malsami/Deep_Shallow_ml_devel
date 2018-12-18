from torch import nn
import torch.nn.functional as F


# Bidirectional neural network is a recurrent neural network forwards and backwards
class BiRNN(nn.module):
    def __init__(self,input_size, hidden_layers, ):
