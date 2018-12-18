from sklearn.neighbors import KNeighborsClassifier
import logging

class KNN:

    model = None

    def __init__(self, num_neighbors, k_weights, k_algorithm):
        self.model = KNeighborsClassifier(n_neighbors=num_neighbors, weights=k_weights, algorithm=k_algorithm)

    def train(self, x,y):
        # self.model.fit(x,y)
        self.model.fit(x,y)
        logging.info("KNN is training")

    def analyze(self):
        pass

    def optimize(self):
        pass
