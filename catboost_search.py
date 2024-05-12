import numpy as np
import optuna
import catboost as cb
from sklearn.metrics import mean_absolute_error,root_mean_squared_error, mean_absolute_percentage_error


def objective(trial, x_train, x_test, y_train, y_test):
    params = {
        "cat_features": ["make", "model", "trim", "body", "transmission", "color", "interior", "seller", "saledate_day", "saledate_month", "saledate_year"],
        "verbose": 0,
        "iterations": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("depth", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        "task_type": "CPU",
        "thread_count": 8,
    }
    cbt = cb.CatBoostRegressor(**params)
    cbt.fit(
        x_train,
        y_train,
        eval_set=(x_test, y_test),
        use_best_model=True,
        early_stopping_rounds=2,
        verbose=False
    )
    y_pred = cbt.predict(x_test)
    score = root_mean_squared_error(y_test, y_pred)
    return score


def getHyperParameter(x_train, x_test, y_train, y_test):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, x_train, x_test, y_train, y_test), n_trials=30)
    print("Number of finished trials: {}".format(len(study.trials)))
    trial = study.best_trial
    return trial.params

