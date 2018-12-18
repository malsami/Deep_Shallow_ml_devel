from torch import nn
import torch.nn.functional as F


class FFN(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, num_classes):
        super().__init__()
        self.h1 = nn.Linear(input_size, hidden_size_1)
        self.h2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.h3 = nn.Linear(hidden_size_2, num_classes)

    def forward(self, x):
        x = self.h1(x)
        x = F.leaky_relu(x)
        x = self.h2(x)
        x = F.leaky_relu(x)
        x = self.h3(x)
        x = F.sigmoid(x)
        return x
