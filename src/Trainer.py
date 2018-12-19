import pickle
import logging
import torch
from models.DeepLearning.FFN import FFN
from models.ShallowLearning.KNearestNeighbors import KNN
from sklearn.model_selection import train_test_split

# Efficient Data Loading
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.utils.data as data_utils

path = 'C:\\Users\\Varun\\Documents\\Misc\\Research\\MalSami\\'

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

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


def torch_model_trainer(x,y, batch_size):
    """TODO ADD RECURRENT NEURAL NETWORK BY INDIAN STUDENT"""

    input_size = x.shape[1]
    hidden_layer_one = 16
    hidden_layer_two = 24
    output_layer = 1

    tf_model = FFN(input_size,hidden_layer_one,hidden_layer_two,output_layer) # .cuda()

    tf_model.cuda()
    logging.info("Model created and using CUDA support")


def torch_optimizer(x,y,batch_size = 64):
    """
    Class that will tune the model via training and validation data. This model makes use of the solver model (taken
    from TUM i2DL/Stanford cs231n class in which a standarad model is used to tune and effectively fit the network
    effectively


    Parameters
    ----------
    x : numpy array [N x D]
        training set samples where n is number of samples and d is the dimension of each sample

    y : numpy array [N x 1]
        training set labels where each sample is either '1' or '0'.

    batch_size: int
        batch_size of the data. Recommendations are (32,64,128..) the most you can to max out the power of the GPU

    """
    # DataLoader for efficient data loading and easy batching
    train_tensor = data_utils.TensorDataset(x,y)

    dataloader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True, num_workers=8)

    logging.info("Data model loaded and ready for hyperparameter tuning")
    # Put into the solver module


def tensorflow_trainer(x,y):

    pass


if __name__ == "__main__":
    logging.basicConfig(filename=path + "Deep_Shallow_ml_devel\\reports\\ml.log", level=logging.INFO)
    logging.info("Logger started")
    x_tensor, x, y_tensor, y = load_data()
    # train_model(x, y, .01)
    torch_model_trainer(x_tensor,y_tensor)

