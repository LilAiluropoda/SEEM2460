import preprocessing as pp
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import message as msg
from dython.nominal import identify_nominal_columns, associations


def corr_matrix_visualize(df):
    categorical_features = identify_nominal_columns(df)
    complete_correlation = associations(df, filename='complete_correlation.png', figsize=(10, 10))
    df_complete_corr = complete_correlation['corr']
    df_complete_corr.dropna(axis=1, how='all').dropna(axis=0, how='all').style.background_gradient(cmap='coolwarm', axis=None)


def matrix_scatter_visualize():
    sns.set_theme(style="ticks")
    plt.figure(figsize=(2, 2))
    pair_plot = sns.pairplot(df, hue="seller", height=2.5)
    pair_plot.savefig("pair_plot_seller.jpg")
    plt.show()


df = pd.read_csv("./data/car_prices.csv")
msg.message("[INFO] Start preprocessing data...")
df = pp.car_preprocessing(df)

corr_matrix_visualize(df)

