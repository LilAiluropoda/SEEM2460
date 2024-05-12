import preprocessing as pp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import helper
from dython.nominal import identify_nominal_columns, associations


def corr_matrix_visualize(data, filename):
    categorical_features = identify_nominal_columns(data)
    print("categorical features: ", categorical_features)
    complete_correlation = associations(data, filename=f'{filename}.png', figsize=(10, 10))
    df_complete_corr = complete_correlation['corr']
    df_complete_corr.dropna(axis=1, how='all').dropna(axis=0, how='all').style.background_gradient(cmap='coolwarm', axis=None)


def matrix_scatter_visualize():
    sns.set_theme(style="ticks")
    plt.figure(figsize=(2, 2))
    pair_plot = sns.pairplot(df, hue="seller", height=2.5)
    pair_plot.savefig("pair_plot_seller.jpg")
    plt.show()


df = pd.read_csv("./data/car_prices.csv")

# preprocess data
helper.message("[INFO] Start preprocessing data...")
df = pp.car_preprocessing(df)

# visualize correlation matrix
helper.message("[INFO] Start visualizing correlation matrix...")
corr_matrix_visualize(df, "corr_matrix_visualize")
helper.message("[INFO] Visualization completed.")

# drop similar columns, >=0.99
drop_col = ["make"]
df.drop(drop_col, axis="columns", inplace=True)
helper.message(f"[INFO] Dropped col: {drop_col}")

# visualize correlation matrix
helper.message("[INFO] Start visualizing correlation matrix...")
corr_matrix_visualize(df, "corr_matrix_visualize_min")
helper.message("[INFO] Visualization completed.")
