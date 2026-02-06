import warnings
import pandas as pd
from modules.classification_metrics import classification_metrics
from modules.crossval import cv_score
from modules.loader import load_data
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

#This is dead code. Modules not used anywhere at the moment.
#modules_to_reload = [classification_metrics,loader, crossval]
#for m in modules_to_reload:
#   importlib.reload(m)
#from my_packages.imports import *
# import importlib


warnings.filterwarnings('ignore')

df = load_data("data/data.csv")

# Here the object and the number variables are not used anywhere.
object = df.select_dtypes(include='object').columns
number = df.select_dtypes(include='number').columns


df = df.drop(['id','Unnamed: 32'], axis=1)


X = df.drop('diagnosis',axis=1)
y = df['diagnosis']
y = pd.get_dummies(y, drop_first=True, dtype=int)


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)


lr = LogisticRegression(penalty='l2')
lr.fit(X_train_scaled,y_train)
y_pred = lr.predict(X_test_scaled)


lr_metric, lr_cm = classification_metrics('Logistic Regression',y_test,y_pred)


lr_cv, lr_cv_scores = cv_score("Logistic Regression", lr, X_train_scaled, y_train, cv=10, scoring='accuracy')


rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train_scaled,y_train)
y_pred = rfc.predict(X_test_scaled)


rfc_metric, rfc_cm = classification_metrics("Random Forrest Classifier", y_test,y_pred)


results = pd.concat([rfc_metric, lr_metric], ignore_index=True)

rfc_cv, rfc_cv_scores = cv_score("Random Forrest",rfc, X_train_scaled, y_train, cv=10, scoring='accuracy')


pd.concat([rfc_cv_scores,lr_cv_scores], ignore_index=True).sort_values('meanCV', ascending=False)


# Logistic regression in this case does a better job than random forest classifier

param_grid = {
    "penalty" : ['l1','l2','elasticnet'],
    "C": [0.25,0.5,0.75,1.0,1.5,1.75,2.0],
    "solver":['saga'],
    "l1_ratio":[0.1,0.5,0.9]
}
gridsearch = GridSearchCV(estimator=LogisticRegression(), param_grid=param_grid, scoring='accuracy',cv=5)
gridsearch.fit(X_train_scaled,y_train)

best_score = gridsearch.best_score_
best_estimator = gridsearch.best_estimator_

print(f"Best Score:\n{best_score}\n")
print(f"Best Estimator:\n{best_estimator}")


# Final Model (Logistic Regression)

final_classifier = LogisticRegression(C=1.5,l1_ratio=0.1,solver='saga')
final_classifier.fit(X_train_scaled,y_train)
y_pred = final_classifier.predict(X_test_scaled)


final_metric, final_cm = classification_metrics('Logistic Refression Final',y_test,y_pred)

result =pd.concat([lr_metric,rfc_metric, final_metric], ignore_index=True)

print(result)


