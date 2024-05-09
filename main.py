import preprocessing as pp
import model
import pandas as pd
import sys

def message(text):
    print("=" * 50)
    print(text)
    print("=" * 50)
def main():
    # Preprocessing
    df = pd.read_csv("./data/car_prices.csv")
    message("[INFO] Start preprocessing data...")
    df = pp.car_preprocessing(df)
    x_train, x_test, y_train, y_test = pp.generate_dataset(df, "sellingprice", 0.2)
    lgbm = model.LightGBM()
    message("[INFO] Start training (LightGBM)...")
    lgbm.train(x_train, y_train, x_test, y_test)
    message("[INFO] Training completed (LightGBM).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
