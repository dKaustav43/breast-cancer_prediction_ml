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


warnings.filterwarnings('ignore')

df = load_data("data/data.csv")

df = df.drop(['id','Unnamed: 32'], axis=1)

X = df.drop('diagnosis',axis=1)
y = df['diagnosis']
y = pd.get_dummies(y, drop_first=True, dtype=int)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# lr = LogisticRegression(penalty='l2')
# lr.fit(X_train_scaled,y_train)
# y_pred = lr.predict(X_test_scaled)

# rfc = RandomForestClassifier(random_state=42)
# rfc.fit(X_train_scaled,y_train)
# y_pred = rfc.predict(X_test_scaled)

# lr_eval_metric, lr_confusion_matrix = classification_metrics('Logistic Regression',y_test,y_pred)
# rfc_eval_metric, rfc_confusion_matrix = classification_metrics("Random Forrest Classifier", y_test,y_pred)


# lr_cv, lr_cv_scores = cv_score("Logistic Regression", lr, X_train_scaled, y_train, cv=10, scoring='accuracy')
# rfc_cv, rfc_cv_scores = cv_score("Random Forrest",rfc, X_train_scaled, y_train, cv=10, scoring='accuracy')


# # As per experimental_notebook.ipynb, Logistic regression in this case does a better job than random forest classifier

# # Doing a grid search with Logistic regression classifier.
# param_grid = {
#     "penalty" : ['l1','l2','elasticnet'],
#     "C": [0.25,0.5,0.75,1.0,1.5,1.75,2.0],
#     "solver":['saga'],
#     "l1_ratio":[0.1,0.5,0.9]
# }
# gridsearch = GridSearchCV(estimator=LogisticRegression(), param_grid=param_grid, scoring='accuracy',cv=5)
# gridsearch.fit(X_train_scaled,y_train)


# best_estimator = gridsearch.best_estimator_

# print(f"Best Estimator:\n{best_estimator}")


# Final Model (Logistic Regression)

final_classifier = LogisticRegression(C=1.5,l1_ratio=0.1,solver='saga')
final_classifier.fit(X_train_scaled,y_train)
y_pred = final_classifier.predict(X_test_scaled)


final_eval_metrics, final_confusion_matrix = classification_metrics('Logistic Refression Final',y_test,y_pred)

#results variable not used anywhere else. Repeat code here. The following results are dead code. 

# results = pd.concat([rfc_eval_metric, lr_eval_metric], ignore_index=True)
# pd.concat([rfc_cv_scores,lr_cv_scores], ignore_index=True).sort_values('meanCV', ascending=False)

#on this result is printed out. 
# result =pd.concat([lr_eval_metric,rfc_eval_metric, final_eval_metrics], ignore_index=True)

print(final_eval_metrics)
print(final_confusion_matrix)

# print(result)


