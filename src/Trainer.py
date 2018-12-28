import pickle
import logging
import torch
from models.DeepLearning.FFN import FFN

# Shallow Learning
from models.ShallowLearning.ShallowModel import ShallowModel
from models.ShallowLearning.LogisticRegression import LogRegress
from models.ShallowLearning.KNearestNeighbors import KNN
# Efficient Data Loading
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data_utils

from src.Utils import load_data, train_val_test_split

# Tuning
from src.Tuning import Solver

path = 'C:\\Users\\Varun\\Documents\\Misc\\Research\\MalSami\\'

# Splits
train_split = .6
val_split = .2
test_split = .2

assert (train_split + val_split + test_split == 1)



def shallow_model_trainer(model, x, y):
    """
    Class that will tune the model via training and validation data. This model makes use of the solver model (taken
    from TUM i2DL/Stanford cs231n class in which a standarad model is used to tune and effectively fit the network
    effectively


    Parameters
    ----------
    model: Sci-kit learn model
        Sci-kit model to optimize

    x : numpy array [N x D]
        training set samples where n is number of samples and d is the dimension of each sample

    y : numpy array [N x 1]
        training set labels where each sample is either '1' or '0'.

    """

    # Split the data
    x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(x=x, y=y, to_tensor=True)

    # model.train(x_train,y_train)
    # shallow_learning_model.train(x_train, y_train)
    model.optimize(x,y,cv = 5)


def torch_model_trainer(x, y, batch_size=64):
    """TODO ADD RECURRENT NEURAL NETWORK BY INDIAN STUDENT"""

    input_size = x.shape[1]
    hidden_layer_one = 16
    hidden_layer_two = 24
    output_layer = 1

    tf_model = FFN(input_size, hidden_layer_one, hidden_layer_two, output_layer)  # .cuda()

    tf_model.cuda()
    logging.info("Model created and using CUDA support")


def torch_optimizer(model, x, y, batch_size=64):
    """
    Class that will tune the model via training and validation data. This model makes use of the solver model (taken
    from TUM i2DL/Stanford cs231n class in which a standarad model is used to tune and effectively fit the network
    effectively


    Parameters
    ----------
    model: Torch model
        Torch model to optimize

    x : numpy array [N x D]
        training set samples where n is number of samples and d is the dimension of each sample

    y : numpy array [N x 1]
        training set labels where each sample is either '1' or '0'.

    batch_size: int
        batch_size of the data. Recommendations are (32,64,128..) the most you can to max out the power of the GPU

    """
    # Splits are by default .6 train, .2 val, .2 test
    x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(x=x.numpy(), y=y.numpy(), to_tensor=True)

    # For data loading
    train_tensor = data_utils.TensorDataset(x_train, y_train)
    val_tensor = data_utils.TensorDataset(x_val, y_val)

    train_data = DataLoader(train_tensor, batch_size=batch_size, shuffle=True, num_workers=-1)
    val_data = DataLoader(val_tensor, batch_size=batch_size, shuffle=True, num_workers=-1)

    logging.info("Data model loaded and ready for hyperparameter tuning")
    # Put into the solver module
    solver = Solver()

    solver.train(model, train_data, val_data)


def tensorflow_trainer(x, y):
    pass


if __name__ == "__main__":
    logging.basicConfig(filename=path + "Deep_Shallow_ml_devel\\reports\\ml.log", level=logging.INFO)

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    logging.info("Logger started")
    x_tensor, x, y_tensor, y = load_data()
    # train_model(x, y, .01)


    # Shallow Models

    shallow_model = KNN(3)

    shallow_model_trainer(shallow_model, x, y)
