import lightgbm as lgb
from catboost import CatBoostRegressor
import shap


class LightGBM:
    model = None
    train_round = 100
    param = {
        # TODO: Hyperparameter Tuning
        "device_type":"cpu",
        "num_threads":8
    }

    def train(self, x_train, y_train, x_test, y_test):
        train_data = lgb.Dataset(x_train, label=y_train)
        test_data = lgb.Dataset(x_test, label=y_test)
        self.model = lgb.train(
            self.param,
            num_boost_round=self.train_round,
            train_set=train_data,
            valid_sets=[test_data],
            callbacks=[lgb.log_evaluation()]
        )
        return 0

    def feature_report(self, x_train, y_train):
        shap.initjs()
        shap_values = shap.Explainer(self.model)(x_train)
        clust = shap.utils.hclust(x_train, y_train, linkage="single")
        shap.plots.bar(shap_values, clustering=clust, clustering_cutoff=1)
        return 0