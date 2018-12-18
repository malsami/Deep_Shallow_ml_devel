from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils

# Trying out for completeness
class kerasFNN:

    model = None

    def __init__(self):
        model = Sequential()
        model.add(Dense(768, input_dim=x.shape[1], kernel_initializer="uniform",
                        activation="relu"))
        model.add(Dense(384, activation="relu", kernel_initializer="uniform"))
        model.add(Dense(1))
        model.add(Activation("softmax"))

    def train(self,x,y,learning_rate):
        sgd = SGD(lr=0.01)
        self.model.compile(loss="binary_crossentropy", optimizer=sgd,
                      metrics=["accuracy"])
        self.model.fit(x, y, epochs=1, batch_size=128, verbose=1)