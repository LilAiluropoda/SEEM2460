import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from dython.nominal import identify_nominal_columns, associations
from scipy.stats.contingency import chi2_contingency
from sklearn.feature_selection import chi2, f_regression
from pandas.plotting import table
import dataframe_image as dfi

# Visualize correlation matrix
def corr_matrix_visualize(df: pd.DataFrame, filename: str):
    categorical_features = identify_nominal_columns(df)
    # Debug Message
    # print("categorical features: ", categorical_features)
    complete_correlation = associations(df, filename=f'graphs/{filename}.png', figsize=(10, 10))
    df_complete_corr = complete_correlation['corr']
    df_complete_corr.dropna(axis=1, how='all').dropna(axis=0, how='all').style.background_gradient(cmap='coolwarm', axis=None)
    return 0


# Visualize scatterplot matrix (Not Used)
def matrix_scatter_visualize(df: pd.DataFrame):
    sns.set_theme(style="ticks")
    plt.figure(figsize=(2, 2))
    pair_plot = sns.pairplot(df, hue="seller", height=2.5)
    pair_plot.savefig("pair_plot_seller.jpg")
    plt.show()
    return 0


def f_test(df: pd.DataFrame, target: str):
    # Separate Numerical and Categorical Columns
    x = df.drop(columns=[target], axis=1).columns

    # Initialize Result Table
    summary = np.empty((len(x), 3), dtype="object")

    for i, col in enumerate(x, start=len(summary)-len(x)):
        t_stat, pvalue = f_regression(df[[col]], df[target].values.reshape(-1, 1))
        summary[i, :] = [col, t_stat[0], pvalue[0]]
    data = pd.DataFrame(
        data=summary,
        columns=["column", 't-statistic', "p-value"]
    )
    summary = data.sort_values(by="t-statistic", ascending=False)
    df_styled = summary.style.bar("t-statistic").background_gradient("Blues", subset="t-statistic")
    dfi.export(df_styled, 'graphs/f_test.png', table_conversion="selenium")
    return data