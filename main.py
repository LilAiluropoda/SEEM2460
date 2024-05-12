import preprocessing as pp
import model
import pandas as pd
import sys
import helper


def main():
    # Preprocessing
    df = pd.read_csv("./data/car_prices.csv")
    helper.message("[INFO] Start preprocessing data...")
    df, mapper, scaler = pp.car_preprocessing(df)
    x_train, x_test, x_valid, y_train, y_test, y_valid = pp.generate_dataset(df, "sellingprice", 0.2)

    # Train LightGBM
    helper.message("[INFO] Start training (LightGBM)...")
    lgbm = model.LightGBM()
    lgbm.train(x_train, y_train, x_valid, y_valid)
    helper.message("[INFO] Training completed (LightGBM).")

    # Train CatBoost
    helper.message("[INFO] Start training (CatBoost)...")
    cbt = model.CatBoost()
    cbt.train(x_train, y_train, x_valid, y_valid)
    helper.message("[INFO] Training completed (CatBoost).")

    # Evaluate LightGBM performance
    lgbm.eval(x_test, y_test)

    # Evaluate CatBoost performance
    cbt.eval(x_test, y_test)

    # Report LightGBM Feature Importance
    helper.message("[INFO] Generating Feature Report (LightGBM)...")
    lgbm.feature_report(x_train, y_train)

    # Report CatBoost Feature Importance
    helper.message("[INFO] Generating Feature Report (CatBoost)...")
    cbt.feature_report(x_train, y_train)

    helper.message("[INFO] Program end, exiting...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
