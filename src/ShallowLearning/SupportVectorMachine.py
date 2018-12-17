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