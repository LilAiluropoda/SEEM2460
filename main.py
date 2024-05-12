import preprocessing as pp
import model
import pandas as pd
import sys
import helper
import lgbm_search
import optuna

def main():
    # Preprocessing
    df = pd.read_csv("./data/car_prices.csv")
    helper.message("[INFO] Start preprocessing data...")
    df = pp.car_preprocessing(df)
    x_train, x_test, y_train, y_test = pp.generate_dataset(df, "sellingprice", 0.2)
    # Train LightGBM
    helper.message("[INFO] Start training (LightGBM)...")
    lgbm = model.LightGBM()
    lgbm.train(x_train, y_train, x_test, y_test)
    helper.message("[INFO] Training completed (LightGBM).")
    # Report LightGBM Feature Importance
    helper.message("[INFO] Generating Feature Report (LightGBM)...")
    lgbm.feature_report(x_train, y_train)

    # Train CatBoost
    helper.message("[INFO] Start training (CatBoost)...")
    cbt = model.CatBoost()
    cbt.train(x_train, y_train, x_test, y_test)
    helper.message("[INFO] Training completed (CatBoost).")

    # Report CatBoost Feature Importance
    helper.message("[INFO] Generating Feature Report (CatBoost)...")
    cbt.feature_report(x_train, y_train)

    helper.message("[INFO] Program end, exiting...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
