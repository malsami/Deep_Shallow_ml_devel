from sklearn.linear_model import LogisticRegression
from models.ShallowLearning import ShallowModel

import numpy as np


class LogRegress(ShallowModel):
    """ LogisticRegression wrapper function. Child of Shallow Model


    Attributes
    ----------
    model : LogisticRegression() from sci-kit learn
        Logistic Regression model with appropriate parameters. User can adjust if necessary.

    hyperpameters: Dict
        List of available hyperparameters to choose from

    Methods
    -------
    train(x,y)
        fits the model with sci-kit learn's 'fit' funciton
    predict(x)
        predicts values for new data 'x'

    """
    model = None
    hyperparameters = None

    def __init__(self, l_penalty='l2', l_C=1.0, l_solver='liblinear', l_max_iter=1000, multi_class='auto', n_jobs=1):
        super(LogRegress, self).__init__(name="Logistic Regression")
        self.model = LogisticRegression(solver=l_solver, max_iter=l_max_iter, multi_class=multi_class, n_jobs=n_jobs)

    def train(self, x, y):
        super(LogRegress, self).train(x, y)

    def predict(self, x):
        super(LogRegress, self).predict(x)

    def optimize(self, x, y, cv=5, verbose=0):
        penalty = ['l1', 'l2']
        C = np.logspace(0, 4, 10)
        self.hyperparameters = dict(C=C, penalty=penalty)
        bf = super(LogRegress, self).optimize(x, y, cv=cv)
        print("Best Penalty: ", bf.best_estimator_.get_params()['penalty'])
        print("Best C: ", bf.best_estimator_.get_params()['C'])
