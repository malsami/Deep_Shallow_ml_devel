import logging


class Shallow_Model():
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
    model = None  # MultinomialNB()

    def __init__(self, name):
        print("Shallow Model Created: ", name)

    def train(self, x, y):
        self.model.fit(x, y)

    def predict(self, x, log_predict=False):
        if (log_predict):
            return self.model.predict_log_proba()
        else:
            return self.model.predict(x)

    def k_fold_fit(self, x, y):
        """
        K fold cross validation for finding fit of data
        :param x:
        :param y:
        :return:
        """

    def optimize(self):
        pass
