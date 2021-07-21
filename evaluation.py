import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix, \
                            classification_report, \
                            roc_auc_score


def score_regressor(X_train, y_train, X_val, y_val, model):
    """Prints the training and validation score of a regression model.
    
    The score in this case R^2 coefficient, defined as 1 - u/v, where 
    u is the sum over all samples of (y_true - y_pred)^2, and v is the 
    sum over all samples of (y_true - y_mean)^2. 

    The best possible score is 1.0, and scores can be negative, which 
    would indicate that the model is worse than the strategy of always
    predicting the average value.
    
    Parameters
    ----------
    X_train (pandas.core.frame.DataFrame
                       or numpy.ndarray): Training input data.
    y_train (pandas.core.frame.DataFrame
                       or numpy.ndarray): Training output values.
    X_val (pandas.core.frame.DataFrame
                       or numpy.ndarray): Validation input data.
    X_train (pandas.core.frame.DataFrame
                       or numpy.ndarray): Validation output values.
    model (sklearn regressor): Model to score. 

    Returns
    ----------
        None
    """
    train_score = round(model.score(X_train, y_train), 4)
    val_score = round(model.score(X_val, y_val), 4)

    print(f"Training score: {train_score}")
    print(f"Validation score: {val_score}")
    
def print_feature_importances(X_features, model):
    """Prints the name of each feature along with the corresponding
    feature importance for the given model. Works with regressors and
    classifiers from sklearn.
    
    Parameters
    ----------
    X_features (list): Names of features on which the model was trained.
    model (sklearn estimator): Model to score. 

    Returns
    ----------
        None
    """
    importances = [round(100*val,3) for val in model.feature_importances_]
    name_val_pairs = list(zip(X_features, importances))
    
    print('\n'.join([str(item) for item in name_val_pairs]))
    
def plot_regr_performance(X, y_true, model, model_name, margin=0.2, 
                          save=False, path=None):
    """Plots the performance of a regression model by scattering points
    of the form (pred_val, true_val). Also plots the "goal line",
    representing what perfect predictions would look like. 
    
    NOTE: only designed for regression models that predict the log (base
    10) of the wall time. The y values should have already had the log
    function applied when they are passed in as the parameter y_true.
    
    Parameters
    ----------
    X (pandas.core.frame.DataFrame
                       or numpy.ndarray): Input data.
    y_true (numpy.ndarray): True values corresponding to input data.
    model (sklearn regressor): Model to score.
    model_name (str): Name of model (used for figure title).
    margin (float): How far the picture should extend on the left,
        right, top, + bottom past the boundaries of the data itself.
    save (bool): Whether the figure should be saved.
    path (str): Path to save the figure (no default value -- a path
        must be provided).

    Returns
    ----------
        None
    """
    # Setting up figure
    plt.figure()
    ax = plt.gca()
    
    # Creating scatter plot of predictions
    pred = model.predict(X)
    val = y_true
    ax.scatter(pred, val, color='darkslateblue', s=1.4)
    
    # Creating "goal line"
    min_value = min(np.amin(pred), np.amin(val)) - margin
    max_value = max(np.amax(pred), np.amax(val)) + margin
    x = [min_value, max_value]
    plt.plot(x, x, '--', color='lightcoral')

    # Axis labels
    plt.xlabel("Predicted wall time")
    plt.ylabel("Actual wall time")
    
    # Axis ticks
    tick_numbers = np.array([60, 120, 300, 900, 3600, 10800, 43200])
    tick_range = np.log10(tick_numbers)
    tick_labels = ["1 min", "2 min", "5 min", "15 min", 
                   "1 hour", "3 hours", "12 hours"]
    plt.xticks(tick_range, labels=tick_labels)
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')

    plt.yticks(tick_range, labels=tick_labels)
    
    # Title
    plt.title(f"Performance Plot: {model_name}")

    if save:
        plt.savefig(path, dpi=120)
    
def plot_cm(X, y_true, model, model_name, save=False, path=None):
    """Plots the confusion matrix for a classification model. 
    
    Note that the displayed percentages are taken over each row.
    Quick example: If the possible classes are 0, 1, and 2, and
    row 1 reads [0.30, 0.60, 0.10], that means that when the 
    correct class was 1, the model guessed class 0 30% of the time,
    class 1 60% of the time, and class 2 10% of the time.
    
    Parameters
    ----------
    X (pandas.core.frame.DataFrame
                       or numpy.ndarray): Input data.
    y_true (numpy.ndarray): True values corresponding to input data.
    model (sklearn regressor): Model to score.
    model_name (str): Name of model (used for figure title).
    save (bool): Whether the figure should be saved.
    path (str): Path to save the figure (no default value -- a path
        must be provided).

    Returns
    ----------
        None
    """
    plt.figure()
    plot_confusion_matrix(model, X, 
                          y_true, normalize='true', 
                          values_format='.2f')
    plt.title(f"Confusion matrix: {model_name}")
    if save:
        plt.savefig(path, dpi=120)

def print_cr(X, y_true, model):
    """Prints the classification report for a classification model. 
    
    Here is an example illustrating the statistics shown.
    Consider the row corresponding to category 2:

    Precision = what percentage of the entries for which the model 
    guessed category 2 were actually in category 2.

    Recall = what percentage of the entries which were in category 2
    were guessed as category 2 by the model.

    f1-Score = a weighted average of precision (P) and recall (R) 
    for category 2, specifically (2PR)/(P+R). Ranges from 0.0 (worst) to 1.0 (best).

    Support = how many entries belonged to category 2.
    
    Parameters
    ----------
    X (pandas.core.frame.DataFrame
                       or numpy.ndarray): Input data.
    y_true (numpy.ndarray): True values corresponding to input data.
    model (sklearn regressor): Model to score.

    Returns
    ----------
        None
    """
    print(classification_report(y_true, model.predict(X)))


def auc(X, y_true, model):
    """Prints the ROC AUC score for a classification model,
    that is, the area under the curve (AUC) of the Receiver Operating
    Characteristic curver (ROC). 

    This is a weighted average of ROC AUC scores for each individual category.
    The ROC AUC score for one category, say category 5 for example, 
    represents the probability that when the model is given two random 
    entries, one belonging to category 5 and one not, the model will 
    assign a higher probability of being in category 5 to the entry that 
    is actually in category 5. 

    Hence the ROC AUC is always a score between 0.0 (worst) and 1.0 (best). 
    
    Parameters
    ----------
    X (pandas.core.frame.DataFrame
                       or numpy.ndarray): Input data.
    y_true (numpy.ndarray): True values corresponding to input data.
    model (sklearn regressor): Model to score.

    Returns
    ----------
        None
    """
    print(round(roc_auc_score(y_true, 
                        model.predict_proba(X),
                        average='weighted',
                        multi_class='ovr'), 4))
    
