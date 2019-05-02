import logging

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


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
    name = ""

    def __init__(self, name):
        self.name = name
        logging.info("Shallow Model Created: %s", name)
        print(name, " created")

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

    def optimize(self, x, y, cv, num_jobs = 32, verbosity=100, exhaustive = False):
        """Grid Search for finding best hyperparameters. WARNING. This may take a long while and may not be optimal.

        Parameters
        ----------
        x : numpy array [N x D]
            training set samples where n is number of samples and d is the dimension of each sample

        y : numpy array [N x 1]
            training set labels where each sample is either '1' or '0'.

        cv: int
            Number of folds for k fold validation

        num_jobs: int optional
            The Number of jobs to run at the same time (parallelism)

        verbosity: int optional
            Whether or not we want to display the gridsearch. The bigger the number, the more output gets displayed. >50 to display output to 
            stdout
        
        exhaustive: boolean optional
            Whether or not we want to do a full Gridsearch or a random gridsearch. Random is defaulted as a recent papers shows evidence for it being
            more effective and efficient than a full gridsearch. 
        """

        # Exhaustive search is for full grid search. Otherwise use a random grid search. A recent paper 
        # shows that this is pretty good. 
        if(exhaustive):
            print("Using Exhaustive Grid Search CV")
            glf = GridSearchCV(self.model, self.hyperparameters, n_jobs = num_jobs, cv=cv, verbose=verbosity)
        else:
            print("Using Randomized Search CV")
            glf = RandomizedSearchCV(estimator = self.model, param_distributions = self.hyperparameters, n_iter=75, cv = cv, 
            verbose = verbosity, random_state=41, n_jobs=num_jobs)

        best_model = glf.fit(x, y)

        return best_model
