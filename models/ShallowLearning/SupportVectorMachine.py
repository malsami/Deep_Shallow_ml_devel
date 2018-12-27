from sklearn import svm

from models.ShallowLearning import ShallowModel

svm_tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
scores = ['precision', 'recall']

"""Grid Search SVM """

parameter_candidates = [
    {'C': [1, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
     'gamma': [1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-3], 'kernel': ['rbf']},
    {'C': [1, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
     'gamma': [1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-3],
     'degree': [1, 2, 3],
     'coef0': [1, 2, 3],
     'kernel': ['poly']},
    {'coef0': [1, 2, 3],
     'gamma': [1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-3],
     'C': [1, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
     'kernel': ['sigmoid']}
]


class SVM(ShallowModel):
    model = None
    hyperparameters = None

    def __init__(self, C_Penalty, s_kernel, s_degree, s_kernel_cache_size, s_tolerance, s_class_weight, s_gamma
                 ):
        super(SVM, self).__init__(self, name="Support Vector Machine")
        self.model = svm.SVC()

    def train(self, x, y):
        super(SVM, self).train(x, y)

    def predict(self, x):
        return super(SVM, self).predict(x)

    def optimize(self, x, y, cv):
        kernel = ['rbf']
        C = [1, 10, 100, 1000]
        degree = [3, 4, 5]
        self.hyperparameters = dict(kernel=kernel, C=C, degree=degree)
        super(SVM, self).optimize(x, y, cv)
