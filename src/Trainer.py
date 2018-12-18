import pickle
import logging
import torch
from models.DeepLearning.FFN import FFN
from models.ShallowLearning.KNearestNeighbors import KNN
from sklearn.model_selection import train_test_split


path = 'C:\\Users\\Varun\\Documents\\Misc\\Research\\MalSami\\'


def load_data():
    # For Deep Learning
    data_tensor = pickle.load(open("x_tensor.p", "rb"))
    labels_tensor = pickle.load(open("y_tensor.p", "rb"))

    # Numpy array for sci-kit learn
    data = data_tensor.numpy()
    labels = labels_tensor.numpy()
    logging.info("Stored data is loaded ")

    return data_tensor, data, labels_tensor, labels


def train_model(x,y,split_size):
    # Split the data
    x_train, x_test, y_train,y_test = train_test_split(x, y, test_size=split_size, random_state=42)

    shallow_learning_model = KNN(3,'uniform','auto')

    shallow_learning_model.train(x,y)


if __name__ == "__main__":
    logging.basicConfig(filename=path + "Deep_Shallow_ml_devel\\reports\\ml.log", level=logging.INFO)
    logging.info("Logger started")
    x_tensor, x, y_tensor, y = load_data()
    train_model(x, y, .01)

