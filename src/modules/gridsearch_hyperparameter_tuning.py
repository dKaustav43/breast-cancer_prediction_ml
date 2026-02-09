import warnings
import pandas as pd
from modules.classification_metrics import classification_metrics
from modules.loader import load_data
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

def gridsearch(estimator, param_grid:dict, scoring:str, cv:int):
    
    df = load_data("data/data.csv")

    df = df.drop(['id','Unnamed: 32'], axis=1)

    X = df.drop('diagnosis',axis=1)
    y = df['diagnosis']
    y = pd.get_dummies(y, drop_first=True, dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)

    gridsearch = GridSearchCV(estimator, param_grid, scoring=scoring, cv=cv)
    gridsearch.fit(X_train_scaled,y_train)

    best_estimator = gridsearch.best_estimator_

    return best_estimator


def main():

    param_grid = {
        "penalty" : ['l1','l2','elasticnet'],
        "C": [0.25,0.5,0.75,1.0,1.5,1.75,2.0],
        "solver":['saga'],
        "l1_ratio":[0.1,0.5,0.9]
    }
    best_estimator = gridsearch(LogisticRegression(),param_grid,scoring='accuracy',cv=5)    
    print(f"Best Estimator:\n{best_estimator}")    

if __name__ == "__main__":
    main()


    

