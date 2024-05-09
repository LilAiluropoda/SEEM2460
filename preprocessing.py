import pandas as pd
import os

def transform_categorical(df, cols):
    mapper = {}
    for col in cols:
        mapped_col, tmp_map = pd.factorize(df[col])
        mapper[col] = tmp_map
        #index starts at 1
        df[col] = pd.Categorical(mapped_col + 1)
    return df, mapper

def transform_datetime(df, col):
    tmp = col + "_utc"
    df[col + "_utc"] = pd.to_datetime(df[col], utc=True)
    df[col + "_year"] = df[tmp].dt.year
    df[col + "_month"] = df[tmp].dt.month
    df[col + "_day"] = df[tmp].dt.day

    # Remove saledate columns
    df.drop(columns=[col, tmp], inplace=True)
    return df

def car_preprocessing(df):
    # Remove rows with missing values
    df.dropna(inplace=True)
    # Remove vin identifier, state
    df.drop(columns=["vin", "state"], inplace=True)
    # Split saledate into year, month, and day
    df = transform_datetime(df, "saledate")
    # Transform classes into categorical variables
    df, mapper = transform_categorical(df, ["make", "model", "trim", "body", "transmission", "color", "interior", "seller"])
    return df

