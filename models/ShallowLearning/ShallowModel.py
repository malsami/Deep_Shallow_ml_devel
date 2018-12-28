import logging

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


class ShallowModel:
    """
    Parent (abstract) class for shallow learning models



    Attributes
    ----------
    model : Any sci-kit learn algorithm
        This model is nothing but an abstract representation of the child classes

    Methods
    -------
    train(x,y)
        fits the model with sci-kit learn's 'fit' funciton
    predict(x)
        predicts values for new data 'x'

    """
    model = None
    hyperparameters = None

    def __init__(self, name):
        logging.info("Shallow Model Created: %s", name)

    def train(self, x, y):
        self.model.fit(x, y)

    def predict(self, x, log_predict=False):
        if log_predict:
            return self.model.predict_log_proba()
        else:
            return self.model.predict(x)

    def validate(self, x, y, num_folds=10):
        """[K fold] cross validation for evaluating the hyperparameters
        Parameters
        ----------

        x: Array [N x D]
            Training samples
        y: Array [N]
            Training labels
        num_folds: int
            Number of folds for cross validation, 10 or fewer is recommended

        """

        kfold = KFold(num_folds, True)

        for train, test in kfold.split(data):
            print('train: %s, test: %s' % (data[train], data[test]))

    def optimize(self, x, y, cv, verbose=0):
        """Grid Search for finding best hyperparameters. WARNING. This may take a long while and may not be optimal.

        Parameters
        ----------
        x : numpy array [N x D]
            training set samples where n is number of samples and d is the dimension of each sample

        y : numpy array [N x 1]
            training set labels where each sample is either '1' or '0'.

        cv: int
            Number of folds for k fold validation

        verbose: int optional
            Whether or not we want to display the gridsearch. '1' for yes, '0' for no.
        """

        glf = GridSearchCV(self.model, self.hyperparameters, cv=cv)
        best_model = glf.fit(x, y)

        return best_model
