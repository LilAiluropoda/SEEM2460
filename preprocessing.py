import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler


def transform_categorical(df, cols: list[str]):
    mapper = {}
    for col in cols:
        mapped_col, tmp_map = pd.factorize(df[col])
        mapper[col] = tmp_map
        # index starts at 1
        df[col] = pd.Categorical(mapped_col + 1)
    return df, mapper


def transform_datetime(df: pd.DataFrame, col: str):
    tmp = col + "_utc"
    df[col + "_utc"] = pd.to_datetime(df[col], utc=True)
    df[col + "_year"] = df[tmp].dt.year
    df[col + "_month"] = df[tmp].dt.month
    df[col + "_day"] = df[tmp].dt.day

    # Set date as categorical variable
    df[col + "_year"] = df[col + "_year"].astype("category")
    df[col + "_month"] = df[col + "_month"].astype("category")
    df[col + "_day"] = df[col + "_day"].astype("category")

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
    # Transform specified columns into categorical variables
    # Transform specified columns into categorical variables
    df, mapper = transform_categorical(df, ["make", "model", "trim", "body", "transmission", "color", "interior", "seller"])
    # Transform specified columns into scaled numerical variables
    df, scaler = transform_scaling(df, ["year", "condition", "odometer", "mmr", "sellingprice"])
    # Transform specified columns into scaled numerical variables
    df, scaler = transform_scaling(df, ["year", "condition", "odometer", "mmr", "sellingprice"])
    df.info()
    return df, mapper, scaler


def transform_scaling(df: pd.DataFrame, cols: list[str]):
    scaler = {}
    for col in cols:
        # Create scaler for target columns
        scaler[col] = MinMaxScaler().fit(df[col].values.reshape(-1, 1))
        # Scale data for target columns
        df[col] = scaler[col].transform(df[col].values.reshape(-1, 1))
    return df, scaler


def generate_dataset(df: pd.DataFrame, target: str, test_size: float):
    x = df.drop(columns=[target], axis=1)
    y = df[target]
    # Split training set and testing set
    x_tmp, x_test, y_tmp, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=314
    )

    # Split training set and validating set
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_tmp,
        y_tmp,
        test_size=test_size,
        random_state=314
    )
    return x_train, x_test, x_valid, y_train, y_test, y_valid

