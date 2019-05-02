import numpy as np 

from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support





import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp

# Time testing analysis 
import time 
from sklearn.model_selection import LeaveOneOut


def classification_time(model, x, y):
    """
    Evaluates the time to classify new samples. As we do not care about the training time, we make 
    this easiesr by doing a LOOCV evaluation. This will simply train the model on N-1 samples and test on the remaining
    1 sample. The algorithm averages the 'error' across all of this. 

    In our scenario, we will only average the time it takes to classify each sample and then average it. 

    Parameters
    ----------
    model : object

        Trained machine learning model we are evaluating

    x : numpy array
        
        Full dataset of taskset data with parameters

    y: numpy array

        Full dataset of labels

    Returns
    -------
    average_classification_time: float
        Average classification time
    """

    # Borrowed from sklearn library 
    running_sum = 0
    loocv = LeaveOneOut()

    for train_index, test_index in loocv.split(x):
    # print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # We are using a trained model, so we can easily just train like this. 
        # model.train(x_train,y_train)
            
        start = time.time()
        model.predict(x_test)
        end = time.time()

    running_sum = running_sum + (end - start)  
    
    # Am Ende
    average_classification_time = running_sum / len(x)

    return average_classification_time

# Taken from sci-kit learn
def plot_confusion_matrix_wrapper(shallow_model, x, y, cv_model, normalize = True, title = None, cmap = plt.cm.Blues):

    """
    Generates the ROC curve for evaluation purposes. Mostly taken from sci-kit learn tutorials

    Parameters
    ----------
    shallow_model: object

        Shallow learning wrappling class model

    x : numpy array [N x D]

        Data

    y : numpy array [N x D]

        Data labels

    cv_model : object

        (K-fold) Cross validator

    """

    np.set_printoptions(precision=2)

    y_pred = cross_val_predict(shallow_model.model, x, y, cv = cv_model)

    unique_structures = unique_labels(y_pred)
    
    
    
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y, y_pred)
    # Only use the labels that appear in the data
    classes = unique_structures
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax



def eval_roc_curve(shallow_model, x,y,validator):
    """
    Generates the ROC curve for evaluation purposes. Mostly taken from sci-kit learn tutorials

    Parameters
    ----------
    shallow_model: object

        Shallow learning wrappling class model

    x : numpy array [N x D]

        Data

    y : numpy array [N x D]

        Data labels

    validator : object

        (K-fold) Cross validator

    """

    precisions = []
    recalls = []
    fscores = []
    supports = []
    y_preds = []
    probs = []

    # AOC from sklearn example
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 0 

    # K fold cross validation
    for train_idx, test_idx in validator.split(x,y):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        shallow_model.model.fit(x_train,y_train)
        y_pred = shallow_model.model.predict(x_test)
        y_pred_probs = shallow_model.model.predict_proba(x_test)

      
        precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred)

        precisions.append(precision)
        recalls.append(recall)
        fscores.append(fscore)
        supports.append(support)


        probs = y_pred_probs[:, 1]
  
        fpr, tpr, thresholds = roc_curve(y_test, probs)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)

        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(shallow_model.name + ' ROC curve')
    plt.legend(loc="lower right")
    # plt.savefig(save_results_to + 'mondorianForestROC.PNG', bbox_inches='tight')
    plt.show()

    # '0' labels
    neg_precision = sum(neg_prec[0] for neg_prec in precisions) / len(precisions)
    neg_recall = sum(neg_rec[0] for neg_rec in recalls) / len(recalls)
    neg_support = sum(supp[0] for supp in supports) / len(supports)
    # '1' labels
    pos_precision = sum(pos_prec[1] for pos_prec in precisions) / len(precisions)
    pos_recall = sum(pos_rec[1] for pos_rec in recalls) / len(recalls)
    pos_support = sum(pos_supp[1] for pos_supp in supports) / len(supports)

    print("average positive (SUCCESSFUL TASKSET) recall: %0.3f" % pos_recall)
    print("average negative (FAIL TASKSET) recall: %0.3f" % neg_recall)
    print("average positive (SUCCESSFUL TASKSET) precision: %0.3f" % pos_precision)
    print("average negative (FAIL TASKSET) precision: %0.3f" % neg_precision)
    print("average support (SUCCESSFUL TASKSET): %1d" % pos_support)
    print("average support (FAIL TASKSET): %1d" % neg_support)



