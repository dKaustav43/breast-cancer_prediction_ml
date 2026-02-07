import warnings
import pandas as pd
from modules.classification_metrics import classification_metrics
from modules.loader import load_data
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


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

Logistic_Regression = LogisticRegression(C=1.5,l1_ratio=0.1,solver='saga')
Logistic_Regression.fit(X_train_scaled,y_train)
y_pred = Logistic_Regression.predict(X_test_scaled)

final_eval_metrics, final_confusion_matrix = classification_metrics('Logistic Refression Final',y_test,y_pred)

print(f"Eval metrics: \n {final_eval_metrics}\n")
print(f"Confusion metrics: \n {final_confusion_matrix}")



