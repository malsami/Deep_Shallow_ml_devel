from torch import nn
import torch.nn.functional as F

<<<<<<< HEAD
"""TODO Add Batch Normalization?"""

=======
>>>>>>> d7e7dcdc28bf4131f091e9843c854b44829b32c1

class FFN(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, num_classes):
        """
        Class for Troch FFN implemented by Robert during Bachelor's thesis. Initiation function for graph.

        ...

        Attributes
        ----------
        input_size: int

            First layer of the neural network corresponds to the size (dimension) of the training data

        hidden_size_1: int

            Hidden layer size chosen by user

        hidden_size_2: int

            Hidden layer size chosen by user

        num_classes: int

            Number of classes will determine the final layer and respective activation function

        Methods
        -------

        forward(x)
            Forward pass of torch model. Will define how data is propogated and computed on in the graph.
        """
        super().__init__()
        self.h1 = nn.Linear(input_size, hidden_size_1)
        self.h2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.h3 = nn.Linear(hidden_size_2, num_classes)

    def forward(self, x):
        """
        Forward pass for pytorch model which computes on x (training data) as it is propogated through network.


        Parameters
        ----------
        x : numpy array [N x D]
            training set samples where n is number of samples and d is the dimension of each sample

        y : numpy array [N x 1]
            training set labels where each sample is either '1' or '0'.

<<<<<<< HEAD
        Returns
        -------
        x: int
            Classification result '1' or '0'
=======
>>>>>>> d7e7dcdc28bf4131f091e9843c854b44829b32c1
        """

        x = self.h1(x)
        x = F.leaky_relu(x)
        x = self.h2(x)
        x = F.leaky_relu(x)
        x = self.h3(x)
        x = F.sigmoid(x)
        return x
