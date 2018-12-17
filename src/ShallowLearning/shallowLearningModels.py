from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import svm







def train_SVM_classifier(x,y):

    # Fit basic SVM and get a feel for the results, then we can add the others
    clf = GridSearchCV(estimator=SVC(), param_grid=parameter_candidates)



def train_logistic_Regression(x,y):
    pass


def train_random_forest(x,y):
    pass


def train_naive_bayes(x,y):
    pass 





