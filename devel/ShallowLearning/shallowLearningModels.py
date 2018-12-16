from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import svm

svm_tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
scores = ['precision', 'recall']

"""Grid Search SVM """


parameter_candidates = [
    {'C': [1, 10, 100, 200,300,400,500,600,700,800,900,1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 200,300,400,500,600,700,800,900,1000],
     'gamma': [1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-3], 'kernel': ['rbf']},
    {'C': [1, 10, 100, 200,300,400,500,600,700,800,900,1000],
     'gamma': [1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-3],
     'degree': [1,2,3],
     'coef0':[1,2,3],
    'kernel': ['poly']},
    {'coef0':[1,2,3],
     'gamma': [1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-3],
     'C': [1, 10, 100, 200,300,400,500,600,700,800,900,1000],
     'kernel': ['sigmoid']}
]

class SVM:

    def __init__(self, C_Penalty,s_kernel,s_degree,s_kernel_cache_size,s_tolerance,s_class_weight,s_gamma
                 ):

        svm.SVC(C = C_Penalty,kernel = s_kernel,degree = s_degree, cache_size = s_kernel_cache_size, tol = s_tolerance,
                class_weight = s_class_weight,gamma=s_gamma
                )
class LogisticRegression:

    def __init__(self):
        pass


class RandomForst:

    def __init__(self):
        pass


def train_SVM_classifier(x,y):

    # Fit basic SVM and get a feel for the results, then we can add the others
    clf = GridSearchCV(estimator=SVC(), param_grid=parameter_candidates)

    svm.SVC(kernel=mkernel, degree=mdegree, cache_size=mcache_size, tol=mtol, class_weight=mclass_weight, C=mC,
            gamma=mgamma)

def train_logistic_Regression(x,y):
    pass


def train_random_forest(x,y):
    pass


def train_naive_bayes(x,y):






