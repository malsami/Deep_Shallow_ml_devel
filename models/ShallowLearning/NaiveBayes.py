# Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

from models.ShallowLearning import Shallow_Model


class GaussianNaiveBayes(Shallow_Model):
    """
    GaussianNB wrapper function. Child of Shallow Model



    Attributes
    ----------
    model : GaussianNB from sci-kit learn
        Naive Bayes with normal/gaussian distribution

    Methods
    -------
    train(x,y)
        fits the model with sci-kit learn's 'fit' funciton
    predict(x)
        predicts values for new data 'x'

    """
    model = None

    def __init__(self):
        super(GaussianNaiveBayes, self).__init__(name="Gaussian Bayes")
        self.model = GaussianNB()

    def train(self, x, y):
        super(GaussianNaiveBayes, self).train(x, y)

    def predict(self, x):
        return super(GaussianNaiveBayes, self).predict(x)


class MultinomialNaiveBayes(Shallow_Model):
    """
    MultinomialNB wrapper function. Child of Shallow Model



    Attributes
    ----------
    model : MultinomialNB from sci-kit learn
        Multinomial Bayes with multinomial gaussian predictions

    Methods
    -------
    train(x,y)
        fits the model with sci-kit learn's 'fit' funciton
    predict(x)
        predicts values for new data 'x'

    """
    model = None

    def __init__(self):
        super(MultinomialNaiveBayes, self).__init__(name="Multinomial Bayes")
        self.model = MultinomialNB()

    def train(self, x, y):
        super(MultinomialNaiveBayes, self).train(x, y)

    def predict(self, x):
        return super(MultinomialNaiveBayes, self).predict(x)


class BernoulliNaiveBayes(Shallow_Model):
    """
        BernoulliNB wrapper function. Child of Shallow Model



    Attributes
    ----------
    model : BernoulliNB from sci-kit learn
        Bernoulli Bayes with multinomial gaussian predictions

    Methods
    -------
    train(x,y)
        fits the model with sci-kit learn's 'fit' funciton
    predict(x)
        predicts values for new data 'x'

    """
    model = None

    def __init__(self):
        super(BernoulliNaiveBayes, self).__init__(name="BernoulliNaiveBayes")
        self.model = BernoulliNB()

    def train(self, x, y):
        super(BernoulliNaiveBayes, self).train(x, y)

    def predict(self, x):
        return super(BernoulliNaiveBayes, self).predict(x)
