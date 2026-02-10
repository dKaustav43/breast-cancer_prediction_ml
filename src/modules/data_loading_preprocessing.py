import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data_remove_columns_splitting_and_scaling_X_y_data(path:str="data/data.csv"):
    
    df = pd.read_csv(path)

    df = df.drop(['id','Unnamed: 32'], axis=1)
    
    X = df.drop('diagnosis',axis=1)
    y = df['diagnosis']
    y = pd.get_dummies(y, drop_first=True, dtype=int)
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
    
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    return X_train_scaled, y_train, X_test_scaled, y_test 



