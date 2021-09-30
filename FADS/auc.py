from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

def calculate_auc(func_model, X_test, y_test, return_pr=False):
    """model: model with .predict or .predict_proba method (duck typing ftw)"""
    #get rows with no missing values in X_train and y_label
    nonmissingX = ~np.isnan(X_test).any(axis=1)
    nonmissingY = ~np.isnan(y_test).any(axis=0)
    nonmissing = nonmissingX & nonmissingY

    X = X_test[nonmissing]
    y = y_test[nonmissing]

    y_pred = func_model(X)

    fpr, tpr, thresholds = roc_curve(y, y_pred, pos_label=1)
    if return_pr:
        return fpr, tpr, auc(fpr,tpr)
    else:
        return auc(fpr, tpr)

def plot_roc(func_model, X_test, y_test, label=None):
    fpr, tpr, auc = calculate_auc(func_model, X_test, y_test, return_pr=True)

    fig = plt.Figure()
    if fig.get_axes():
        plt.plot([0, 1], [0, 1], 'k--', label="Random classifier")
    if label!=None:
        plt.plot(fpr, tpr, label=label + "with AUC={:.2f}".format(auc))
    else:
        plt.plot(fpr, tpr, label="AUC={:.2f}".format(auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend()

    return fig


