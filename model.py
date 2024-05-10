import lightgbm as lgb
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import shap
import helper


class LightGBM:
    model = None
    train_round = 100
    param = {
        # TODO: Hyperparameter Tuning
        # Must have, cannot be changed
        "device_type":"cpu",
        "num_threads":8
    }

    def train(self, x_train, y_train, x_test, y_test):
        # Wrapping train and test dataset
        train_data = lgb.Dataset(x_train, label=y_train)
        test_data = lgb.Dataset(x_test, label=y_test)

        # Create model
        self.model = lgb.train(
            self.param,
            num_boost_round=self.train_round,
            train_set=train_data,
            valid_sets=[test_data],
            callbacks=[lgb.log_evaluation()]
        )
        return 0

    def feature_report(self, x_train, y_train):
        # initialize JavaScript Visualization Library
        shap.initjs()
        helper.message("[INFO] Training explainer for LightGBM ...")
        tree = shap.TreeExplainer(self.model).shap_values(x_train)
        helper.message("[INFO] Training completed, visualising...")
        shap.summary_plot(tree, x_train)
        plt.show()
        return 0


class CatBoost:
    model = None
    param = {
        # TODO: Hyperparameter Tuning
        # Must have, cannot be changed
        'cat_features': ["make", "model", "trim", "body", "transmission", "color", "interior", "seller", "saledate_day", "saledate_month", "saledate_year"],
        'verbose': 200
    }

    def train(self, x_train, y_train, x_test, y_test):
        self.model = CatBoostRegressor(**self.param)
        self.model.fit(
            x_train,
            y_train,
            eval_set=(x_test, y_test),
            use_best_model=True
        )
        return 0

    def feature_report(self, x_train, y_train):
        # initialize JavaScript Visualization Library
        shap.initjs()
        helper.message("[INFO] Training explainer for CatBoost ...")
        tree = shap.TreeExplainer(self.model).shap_values(x_train)
        helper.message("[INFO] Training completed, visualising...")
        shap.summary_plot(tree, x_train)
        plt.show()
        return 0
