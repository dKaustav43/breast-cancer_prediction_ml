import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path:str):
    "Load CSV dataset."
    return pd.read_csv(path)

def split_data(df, test_size=0.2, random_state=42):
    "Split into train/test sets."
    return train_test_split(df, test_size, random_state)