import numpy as np
import optuna
import lightgbm as lgb
import sklearn.datasets
import sklearn.metrics

def objective(trial,x_train,x_test,y_train,y_test):
    train_data = lgb.Dataset(x_train, label=y_train)
    param = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "device_type":"cpu",
        "num_threads":8,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    gbm = lgb.train(param, train_data)
    preds = gbm.predict(x_test)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(y_test, pred_labels)
    return accuracy


def getHyperParameter(x_train,x_test,y_train,y_test):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial : objective(trial,x_train,x_test,y_train,y_test), n_trials=100)
    print("Number of finished trials: {}".format(len(study.trials)))
    trial = study.best_trial
    return trial.params

