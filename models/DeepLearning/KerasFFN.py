from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense

"""
    https://stackoverflow.com/questions/45662253/can-i-run-keras-model-on-gpu
    Running keras on GPU 
"""


# Trying out for completeness
class kerasFNN:
    """Class for Keras FFN with tensorflow backend


    Attributes
    ----------
    model : Sequential graph model

    Methods
    -------
    train(x,y,learning_rate = 0.001)
        Implicity compiles graph model and training module and combines for fitting and learning.
    """
    model = None

    def __init__(self, hidden_layer_1=768, hidden_layer_2=364, output_layer=1):
        """
        Parameters
        ----------
        hidden_layer_1 : int
            The number of nodes for the first 'hidden' layer of the network.
            Default to 768
        hidden_layer_2: int
            The number of nodes for the second 'hidden' layer of the network.
            Default to 364
        output_layer : int,
            Default to 1 because of binary classification
        """

        model = Sequential()
        model.add(Dense(768, input_dim=x.shape[1], kernel_initializer="uniform",
                        activation="relu"))
        model.add(Dense(384, activation="relu", kernel_initializer="uniform"))
        model.add(Dense(1))
        model.add(Activation("softmax"))

    def train(self, x, y, learning_rate=0.001):
        """Compiles the static graph and the trainer together and then fits according to the inputted data

        If the learning_rate argument is not provided, then the default 1e-3 is used

        Parameters
        ----------
        x : numpy array [N x D]
            training set samples where n is number of samples and d is the dimension of each sample

        y : numpy array [N x 1]
            training set labels where each sample is either '1' or '0'.


        """
        sgd = SGD(lr=0.01)
        self.model.compile(loss="binary_crossentropy", optimizer=sgd,
                           metrics=["accuracy"])
        self.model.fit(x, y, epochs=1, batch_size=128, verbose=1)
