import pandas as pd
from sklearn.model_selection import train_test_split
import os


def transform_categorical(df, cols):
    mapper = {}
    for col in cols:
        mapped_col, tmp_map = pd.factorize(df[col])
        mapper[col] = tmp_map
        #index starts at 1
        df[col] = pd.Categorical(mapped_col + 1)
    return df, mapper


def transform_datetime(df: pd.DataFrame, col: str):
    tmp = col + "_utc"
    df[col + "_utc"] = pd.to_datetime(df[col], utc=True)
    df[col + "_year"] = df[tmp].dt.year
    df[col + "_month"] = df[tmp].dt.month
    df[col + "_day"] = df[tmp].dt.day

    # Remove saledate columns
    df.drop(columns=[col, tmp], inplace=True)
    return df


def car_preprocessing(df: pd.DataFrame):
    # Remove rows with missing values
    df.dropna(inplace=True)
    # Remove vin identifier, state
    df.drop(columns=["vin", "state"], inplace=True)
    # Split saledate into year, month, and day
    df = transform_datetime(df, "saledate")
    # Transform classes into categorical variables
    df, mapper = transform_categorical(df, ["make", "model", "trim", "body", "transmission", "color", "interior", "seller"])
    # Reduce dataset size
    #df = df.head(100000)
    df.info()
    return df


def generate_dataset(df: pd.DataFrame, target: str, test_size: float):
    x = df.drop(columns=[target], axis=1)
    y = df[target]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=314
    )
    return x_train, x_test, y_train, y_test

