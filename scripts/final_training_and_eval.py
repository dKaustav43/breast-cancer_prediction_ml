import warnings
import pandas as pd
from modules.classification_metrics import classification_metrics
from modules.loader import load_data
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from modules.gridsearch_hyperparameter_tuning import gridsearch_bestestimator


warnings.filterwarnings('ignore')

def training_and_eval(final_model_name:str, model_instance_with_best_params, path_to_data:str):

    df = load_data(path_to_data)
    
    df = df.drop(['id','Unnamed: 32'], axis=1)
    
    X = df.drop('diagnosis',axis=1)
    y = df['diagnosis']
    y = pd.get_dummies(y, drop_first=True, dtype=int)
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
    
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)
    
    model_instance_with_best_params.fit(X_train_scaled,y_train)
    y_pred = model_instance_with_best_params.predict(X_test_scaled)
    
    eval_metrics, confusion_matrix = classification_metrics(final_model_name,y_test,y_pred)
    
    return eval_metrics, confusion_matrix


def main():
    
    eval_metrics, confusion_matrix = training_and_eval(final_model_name = 'Logistic Regression', model_instance_with_best_params=gridsearch_bestestimator(), path_to_data = "data/data.csv")

    print(f"Eval metrics: \n {eval_metrics}\n")
    print(f"Confusion metrics: \n {confusion_matrix}")

if __name__ == "__main__":
    main()


