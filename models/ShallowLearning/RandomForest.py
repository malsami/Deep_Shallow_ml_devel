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

    def __init__(self, rf_estimators=1000, rf_max_depth=70, rf_n_jobs=-1, rf_min_samples_leaf = 3,
                rf_min_samples_split = 4, rf_max_features = 'auto'
                ):
        super(RandomForest, self).__init__(name="Random Forest")
        self.model = RandomForestClassifier(n_estimators=rf_estimators, max_depth=rf_max_depth, n_jobs=rf_n_jobs)
        # logging.basicConfig(filename= path + "reports\\" + "rf.log", level=logging.info)
        logging.info("Rand Forest Log created")

    def train(self, x, y):
        super(RandomForest, self).train(x, y)

    def fit(self, x, y):
        super(RandomForest, self).train(x, y)

    def predict(self, x):
        return super(RandomForest, self).predict(x)

    def optimize(self, x, y, cv = 5):
        
        self.hyperparameters = {
            'bootstrap': [True, False],
            'max_depth': [10,20,30,40,50,60,70,80, 90,100,None],
            'max_features': ['auto', 'sqrt'],
            'min_samples_leaf': [3, 6, 9],
            'min_samples_split': [2, 4,8,16],
            'n_estimators': [150, 250, 350,500,750,1000,1250,1500,2000]
        }

        # Current best
        # Depth: 70
        # Max features: auto
        # Best min_samples_leaf: 3
        # Best min_samples_split: 4
        # n_estimators: 1000

        bf = super(RandomForest, self).optimize(x, y, cv)
        print("Best Depth: ", bf.best_estimator_.get_params()['max_depth'])
        print("Best max_features: ", bf.best_estimator_.get_params()['max_features'])
        print("Best min samples_leaf: ", bf.best_estimator_.get_params()['min_samples_leaf'])
        print("Best min_samples_split: ", bf.best_estimator_.get_params()['min_samples_split'])
        print("Best n_estimators: ", bf.best_estimator_.get_params()["n_estimators"])
        

    def analyze(self):
        pass
