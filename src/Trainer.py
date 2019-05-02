import pickle
import logging
import torch
import sys
sys.path.append("../")

import numpy as np

# Deep Learning
from models.DeepLearning.FFN import FFN
from models.DeepLearning.KerasFFN import kerasFNN
# Shallow Learning
from models.ShallowLearning.ShallowModel import ShallowModel
from models.ShallowLearning.LogisticRegression import LogRegress
from models.ShallowLearning.KNearestNeighbors import KNN
from models.ShallowLearning.RandomForest import RandomForest
from models.ShallowLearning.SupportVectorMachine import SVM
from models.ShallowLearning.MondorianForest import MondorianForest
from models.ShallowLearning.NaiveBayes import GaussianNaiveBayes, MultinomialNaiveBayes, BernoulliNaiveBayes

# Efficient Data Loading
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data_utils

# Validation
from sklearn.model_selection import StratifiedKFold


from Utils import load_data, train_val_test_split, balanced_subsample

# Tuning
#from Tuning.Solver import Solver
# Validation
from sklearn.model_selection import cross_val_score

# Classification Report
from sklearn.metrics import classification_report

from Evaluation import eval_roc_curve, classification_time

import time 



path = '../'

# Splits
train_split = .6
val_split = .2
test_split = .2

assert (train_split + val_split + test_split == 1)


def cross_validation_scoring(shallow_model,x,y, cv = 5, strat_fold = True):
    """
       Class that will use kfold validation for scoring

       Parameters
       ----------
       s_model: Sci-kit learn model
           Sci-kit model to optimize

       x : numpy array [N x D]
           training set samples where n is number of samples and d is the dimension of each sample

       y : numpy array [N x 1]
           training set labels where each sample is either '1' or '0'.

       strat_fold: boolean
            defaulted to true to attempt even splits of samples
        Returns
        ---------
        numpy array [cv x 1]
            scores where each element is the score for the particular 'fold'
       """

    if(strat_fold):
        skf = StratifiedKFold(n_splits=10, shuffle=True)
     
    else:
        pass
        # kkf = KFold(10, True, 41)

    return cross_val_score(shallow_model.model, x, y, cv=cv)


def shallow_model_trainer(s_model, x, y):
    """
    Class that will tune the model via training and validation data. This model makes use of the solver model (taken
    from TUM i2DL/Stanford cs231n class in which a standarad model is used to tune and effectively fit the network
    effectively


    Parameters
    ----------
    s_model: Sci-kit learn model
        Sci-kit model to optimize

    x : numpy array [N x D]
        training set samples where n is number of samples and d is the dimension of each sample

    y : numpy array [N x 1]
        training set labels where each sample is either '1' or '0'.

    """

    # Split the data
    x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(x=x, y=y, to_tensor=True)

    # model.train(x_train,y_train)
    s_model.train(x_train, y_train)
    y_pred = s_model.predict(x_val)
    # s_model.optimize(x,y,cv = 5)

    return s_model, y_pred, y_val


def torch_model_trainer(x, y, batch_size=4):
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

    train_data = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    val_data = DataLoader(val_tensor, batch_size=batch_size, shuffle=True)

    logging.info("Data model loaded and ready for hyperparameter tuning")
    # Put into the solver module
    #solver = Solver()

    #solver.train(model, train_data, val_data)


def tensorflow_trainer(x, y):
    pass


if __name__ == "__main__":
    logging.basicConfig(filename=path + "reports/ml.log", level=logging.INFO)

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    logging.info("Logger started")
    # Tensors for torch, regular for sci-kit learn and keras/tensorflow
    x_tensor, x, y_tensor, y = load_data()
    print("Data Successfully loaded")

    

    # Uncomment for I/O based implementation
    """
    _knn = "(K)-Nearest Neighbor"
    _lr = "(L)ogistic Regression"
    _svm = "(S)upport Vector Machine"
    _rf = "(R)andom Forest"
    _mf = "(M)ondorian Forest"

    while(True):
        learning_selection = input("(S)hallow Learning or (D)eep Learning?")

        if(learning_selection == "S" or learning_selection == "s"):
            model_selection = input("Choose Shallow Model:'\n'
                "")
            break
        if(learning_selection == "D" or learning_selection == "d"):
            model_selection = input("Choose Deep Model'\n'")
            break
    """

    # train_model(x, y, .01)

    # Deep Learning (Torch)
    # deep_model = FFN(input_size=x.shape[1], hidden_size_1=16,hidden_size_2=24,num_classes=2)
    #torch_optimizer(deep_model,x_tensor,y_tensor, batch_size=4)
    #print("Optimization done")
    # Keras/Tensorflow (This has nice front end) works better in Jupyter notebook though
    # deep_model = kerasFNN(input_layer=x.shape[1], hidden_layer_1=768,hidden_layer_2=364,output_layer=1)
    # deep_model.train(x,y)
    


    # Shallow Models


    # Unccomment one at a time
    # shallow_model = KNN(5)
    # shallow_model = MondorianForest()
    # shallow_model = LogRegress()
    # shallow_model = SVM() 
    # shallow_model = RandomForest()
    # shallow_model = GaussianNaiveBayes()
    # shallow_model = MultinomialNaiveBayes()
    shallow_model = BernoulliNaiveBayes()
    #skf_scores = cross_validation_scoring(shallow_model, x, y)

    #print("Stratified 10-fold mean: %0.3f" % np.mean(skf_scores))
    #print("Stratified 10-fold variance: %3f" % np.var(skf_scores))
    # Grid search
    #print("optimizing...")
    # shallow_model.optimize(x,y,cv = 5)

    # For either trainign or optimization
    # print("Training model...")
    # x_s, y_s = balanced_subsample(x,y)

    # trained_model, y_pred, y_val = shallow_model_trainer(shallow_model, x, y)
    # skf = StratifiedKFold(n_splits=5, shuffle=True)
    # skf = StratifiedKFold(n_splits=5, shuffle=True)
    # eval_roc_curve(shallow_model.model,x,y,skf)

    # print(classification_report(y_val, y_pred))


    # Timing analysis 
    x_train,y_train, _,_,x_test,y_test = train_val_test_split(x,y)
    shallow_model.train(x_train, y_train)

    print("Entering the timing classification")
    # average_time = classification_time(shallow_model, x_test, y_test)
    start = time.time()
    shallow_model.predict(x_test)
    end = time.time()



    print("samples: ", len(x_test))
    print("Average classification time: ", (end - start))

   
