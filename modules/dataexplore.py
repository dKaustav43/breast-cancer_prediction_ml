
#Code for basic data exploration, stas and missing values.

from IPython.display import display, Markdown
import pandas as pd

def dataframedatatypes(df:pd.DataFrame):
    
    print(f"Shape: {df.shape}")
    print(f"Column information: {df.info()}")

def columntypes(df:pd.DataFrame):
    
    object = df.select_dtypes(include='object').columns
    number = df.select_dtypes(include='number').columns

    display(Markdown("### Columns with dtype:object"))
    display(object)

    display(Markdown("### Columns with dtype:Number"))
    display(number)

    return pd.DataFrame({'object':object}), pd.DataFrame({'number':number})

def statsummary(df:pd.DataFrame):

    print("Descriptive stats")
    return df.describe().T

def checkmissingvalues(df:pd.DataFrame):
    missing = df.isnull().sum()
    percent = ((missing/len(df))) * 100
    summary = pd.DataFrame({
        'Missing count' : missing,
        "Missing percent": percent
    })
    return summary.sort_values("Missing percent", ascending=False)


