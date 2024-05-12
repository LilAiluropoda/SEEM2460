import numpy as np
import optuna
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error,root_mean_squared_error, mean_absolute_percentage_error
import sklearn.metrics

def objective(trial,x_train,x_test,y_train,y_test):
    train_data = lgb.Dataset(x_train, label=y_train)
    param = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "device_type":"cpu",
        "num_threads":8,
        "n_estimators": trial.suggest_categorical("n_estimators", [1000]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "max_bin": trial.suggest_int("max_bin", 200, 300),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }
    gbm = lgb.train(param, train_data)
    y_pred = gbm.predict(x_test)
    score = root_mean_squared_error(y_test, y_pred)
    return score

def getHyperParameter(x_train,x_test,y_train,y_test):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial : objective(trial,x_train,x_test,y_train,y_test), n_trials=100)
    print("Number of finished trials: {}".format(len(study.trials)))
    trial = study.best_trial
    return trial.params

