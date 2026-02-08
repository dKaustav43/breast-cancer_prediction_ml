import pandas as pd
from sklearn.model_selection import cross_val_score

def cv_score(model:str, model_instance, X_train, y_train, cv:int, scoring:str):
    """
    Computed cross_val_score and returns
    - An array of Cross_Val score
    - Mean cross_val score 
    - Std. dev. of cross_val scores

    Parameters:
        model:str - Name of the model passed as a string
        X_train - array like training data
        y_train - array like true labels

    Returns:
        Mean and Stddev of CV scores (pd.DataFrame)

    """
    cv_score = cross_val_score(model_instance,X=X_train,
                               y=y_train,cv=cv,scoring=scoring).round(3)
    
    mean_score = cv_score.mean().round(3)
    stddev_score = cv_score.std().round(3)
    
    Crossval_mean_stddev_scores = pd.DataFrame([[model, mean_score , stddev_score]], columns=["model","meanCV","stddevCV"])

    return Crossval_mean_stddev_scores