from sklearn.ensemble import RandomForestClassifier

from models.ShallowLearning import ShallowModel


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

    def __init__(self, rf_estimators=15, rf_max_depth=2, rf_n_jobs=-1):
        super(RandomForest, self).__init__(name="Random Forest")
        self.model = RandomForestClassifier(n_estimators=rf_estimators, max_depth=rf_max_depth, n_jobs=rf_n_jobs)

    def train(self, x, y):
        super(RandomForest, self).train(x, y)

    def predict(self,x):
        return super(RandomForest, self).predict(x)

    def optimize(self,x,y,cv):
        hyper_parameters = {
            'bootstrap': [True],
            'max_depth': [80, 90],
            'max_features': [2, 3],
            'min_samples_leaf': [3, 4, 5],
            'min_samples_split': [8, 10],
            'n_estimators': [100, 200, 300]
        }

        super(RandomForest, self).optimize(x, y, cv)


    def analyze(self):
        pass
