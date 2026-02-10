import warnings
import pandas as pd
from modules.crossval import cv_score
from modules.data_loading_preprocessing import load_data_remove_columns_splitting_and_scaling_X_y_data
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings('ignore')

def comparing_models():

    X_train_scaled, y_train, X_test_scaled, y_test = load_data_remove_columns_splitting_and_scaling_X_y_data()

    #Initializing the models.
    lr = LogisticRegression(penalty='l2')
    rfc = RandomForestClassifier(random_state=42)
    
    # Compute cross_val_score for both the models and get the mean and stddev of the scores. Only cross_val tells that lr is better than rfc. The eval metrics are not able to capture the true performance of the model as they are based on a single train-test split.
    lr_crossval_mean_stddev_scores = cv_score("Logistic Regression", lr, X_train_scaled, y_train, cv=10, scoring='accuracy')
    rfc_crossval_mean_stddev_scores = cv_score("Random Forrest",rfc, X_train_scaled, y_train, cv=10, scoring='accuracy')
    
    models_crossval_metrics = pd.concat([lr_crossval_mean_stddev_scores,rfc_crossval_mean_stddev_scores], ignore_index=True).sort_values('meanCV', ascending=False)
    
    return models_crossval_metrics

def main():
    
    crossval_metrics = comparing_models()
    print(f"The crossval metrics of the models are: \n {crossval_metrics}")

if __name__ == "__main__":
    main()



