import preprocessing as pp
import model
import pandas as pd
import sys
import helper


def main():
    # Preprocessing
    df = pd.read_csv("./data/car_prices.csv")
    helper.message("[INFO] Start preprocessing data...")
    df = pp.car_preprocessing(df)
    x_train, x_test, y_train, y_test = pp.generate_dataset(df, "sellingprice", 0.2)
    lgbm = model.LightGBM()
    helper.message("[INFO] Start training (LightGBM)...")
    lgbm.train(x_train, y_train, x_test, y_test)
    helper.message("[INFO] Training completed (LightGBM).")
    helper.message("[INFO] Generating Feature Report (LightGBM)...")
    lgbm.feature_report(x_train, y_train)
    return 0


if __name__ == "__main__":
    sys.exit(main())
