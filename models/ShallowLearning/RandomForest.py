from sklearn.ensemble import RandomForestClassifier

from models.ShallowLearning.ShallowModel import ShallowModel
import logging
from src.Utils import path


class RandomForest(ShallowModel):
    """
    RandomForest wrapper function. Child of Shallow Model. Can be modified to exclude sci-kit learn if developer
    prefers


    Attributes
    ----------
    model : RandomForestClassifier() from sci-kit learn
        Ensemble of trees used from sci-kit learn

    Methods
    -------
    train(x,y)
        fits the model with sci-kit learn's 'fit' funciton
    predict(x)
        predicts values for new data 'x'

    """
    model = None
    hyperparameters = None

    def __init__(self, rf_estimators=15, rf_max_depth=2, rf_n_jobs=-1):
        super(RandomForest, self).__init__(name="Random Forest")
        self.model = RandomForestClassifier(n_estimators=rf_estimators, max_depth=rf_max_depth, n_jobs=rf_n_jobs)
        logging.basicConfig(filename= path + "reports\\" + "rf.log", level=logging.info)
        logging.info("Rand Forest Log created")

    def stub(self):
        x = 5
        logging.info("This is a stub: ", x)

    def train(self, x, y):
        super(RandomForest, self).train(x, y)

    def predict(self, x):
        return super(RandomForest, self).predict(x)

    def optimize(self, x, y, cv):
        self.hyperparameters = {
            'bootstrap': [True],
            'max_depth': [80, 90],
            'max_features': [2, 3],
            'min_samples_leaf': [3, 4, 5],
            'min_samples_split': [8, 10],
            'n_estimators': [100, 200, 300]
        }

        bf = super(RandomForest, self).optimize(x, y, cv)
        print("Best Depth: ", bf.best_estimator_.get_params()['max_depth'])
        print("Best max_features: ", bf.best_estimator_.get_params()['max_features'])
        print("Best min samples_leaf: ", bf.best_estimator_.get_params()['min_samples_leaf'])
        print("Best min_samples_split: ", bf.best_estimator_.get_params()['min_samples_split'])
        print("Best n_estimators: ", bf.best_estimator_.get_params()["n_estimators"])

    def analyze(self):
        pass
