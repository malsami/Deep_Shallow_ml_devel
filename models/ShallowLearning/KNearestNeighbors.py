from sklearn.neighbors import KNeighborsClassifier
import logging

# Super Class
from models.ShallowLearning import ShallowModel


class KNN(ShallowModel):
    model = None

    def __init__(self, num_neighbors, k_weights='uniform', k_algorithm='auto'):
        super(KNN, self).__init__(name="KNN with " + str(num_neighbors) + " neighbors")
        self.model = KNeighborsClassifier(n_neighbors=num_neighbors, weights=k_weights, algorithm=k_algorithm)

    def train(self, x, y):
        self.model.fit(x)
        logging.info("KNN is training")

    def predict(self, x):
        super(KNN, self).predict(x)

    def analyze(self):
        pass

    def optimize(self):
        pass
