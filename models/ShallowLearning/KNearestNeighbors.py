from sklearn.neighbors import KNeighborsClassifier
import logging

# Super Class
from models.ShallowLearning.ShallowModel import ShallowModel


class KNN(ShallowModel):
    model = None
    hyperparameters = None

    def __init__(self, num_neighbors, k_weights='uniform', k_algorithm='auto'):
        super(KNN, self).__init__(name="KNN with " + str(num_neighbors) + " neighbors")
        self.model = KNeighborsClassifier(n_neighbors=num_neighbors, weights=k_weights, algorithm=k_algorithm)

    def train(self, x, y):
        self.model.fit(x,y)
        logging.info("KNN is training")

    def predict(self, x):
        super(KNN, self).predict(x)

    def optimize(self, x, y, cv, verbose=0):
        k_range = list(range(1, 10))
        print("Grid search in progress")
        self.hyperparameters = dict(n_neighbors=k_range)
        bf = super(KNN, self).optimize(x, y, cv=cv)
        print("Best n_neighbors: ", bf.best_estimator_.get_params()["n_neighbors"])

    def analyze(self):
        pass

