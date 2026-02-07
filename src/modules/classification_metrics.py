# import the metrics modules
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# define a function which returns specific metrics, with inputs y_pred, y_test
def classification_metrics(Model:str, y_test,y_pred):
    
    """
    Computes classification metrics and returns:
    - A DataFrame with Accuracy, Precision, Recall, and F1 Score
    - A confusion matrix (as a DataFrame)
    
    Parameters:
        y_test: array-like of true labels
        y_pred: array-like of predicted labels
    
    Returns:
        metrics_df (pd.DataFrame)
        cm_df (pd.DataFrame)
    """

    acc = accuracy_score(y_test,y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test,y_pred)
    eval_metrics_df = pd.DataFrame([[Model, acc, f1, prec, rec]], 
                        columns = ['Model','Accuracy', 'F1 Score', 'Precision', 'Recall']).round(3)
    confusion_matrix_df = pd.DataFrame(cm, index = ['Actual 0','Actual 1'], columns = ['Pred 0','Pred 1'])

    return eval_metrics_df, confusion_matrix_df
