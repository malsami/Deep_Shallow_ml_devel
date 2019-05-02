from sklearn import svm

from models.ShallowLearning.ShallowModel import ShallowModel

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

    def __init__(self, linear = False):
        super(SVM, self).__init__(name="SVM")

        if(linear):
            self.model = svm.LinearSVC()
        else:
            self.model = svm.SVC()
        # svm.SVC(kernel=s_kernel, degree=s_degree, cache_size=s_kernel_cache_size, tol=s_tolerance,
        #                     class_weight=s_class_weight, gamma=s_gamma)

    def train(self, x, y):
        super(SVM, self).train(x, y)

    def predict(self, x):
        return super(SVM, self).predict(x)

    def optimize(self, x, y, cv):
        
        self.hyperparameters = {
            'C': [1, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'gamma' : [1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-3],
            'degree': [1, 2, 3],
            'coef0': [1, 2, 3]
        }
         
        # self.hyperparameters = dict(kernel=kernel, C=C, degree=degree)
        self.hyperparameters = parameter_candidates[0]
        bf = super(SVM, self).optimize(x, y, cv)

        # Current Best
        # C : 1
        # Best kernel: linear
        # gamma deprecated: auto-deprecated
        # degree: 3
        # coef0: 0.0
        print("Best C: ", bf.best_estimator_.get_params()['C'])
        print("Best kernel: ", bf.best_estimator_.get_params()['kernel'])
        print("Best gamma: ", bf.best_estimator_.get_params()['gamma'])
        print("Best degree: ", bf.best_estimator_.get_params()['degree'])
        print("Best coef0: ", bf.best_estimator_.get_params()["coef0"])
