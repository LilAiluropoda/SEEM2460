import lightgbm as lgb
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import shap
import helper
import numpy as np
import lgbm_search
import catboost_search
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error


class LightGBM:
    model = None
    train_loss = {}

    def get_param(self, x_train, x_valid, y_train, y_valid):
        # Use optuna to find the HyperParameter
        helper.message("[INFO] Tuning hyperparameter (LightGBM) ...")
        param = lgbm_search.getHyperParameter(x_train, x_valid, y_train, y_valid)
        
        # Add compulsory hyperparameter
        param["verbosity"] = 1
        param["num_threads"] = 8
        param["device_type"] = "cpu"
        param["early_stopping_round"] = 5
        param["objective"] = "regression"
        param["metric"] = "rmse"
        
        helper.message("[INFO] Tuning completed.")
        helper.message(param)
        return param

    def train(self, x_train, y_train, x_valid, y_valid):
        # Wrapping train and test dataset
        train_data = lgb.Dataset(x_train, label=y_train)
        test_data = lgb.Dataset(x_valid, label=y_valid)

        # Tune hyperparameter
        param = self.get_param(x_train, x_valid, y_train, y_valid)

        # Create model
        helper.message("[INFO] Training model with tuned hyperparameter ... ")
        self.model = lgb.train(
            param,
            num_boost_round=300,
            train_set=train_data,
            valid_sets=[train_data, test_data],
            callbacks=[lgb.log_evaluation(), lgb.record_evaluation(eval_result=self.train_loss)]
        )
        return 0

    def feature_report(self, x_train, y_train):
        # initialize JavaScript Visualization Library
        shap.initjs()
        helper.message("[INFO] Training explainer for LightGBM ...")
        tree = shap.TreeExplainer(self.model).shap_values(x_train)
        helper.message("[INFO] Training completed, visualising...")
        shap.summary_plot(tree, x_train, plot_type="bar",show=False)
        plt.savefig("graphs/lgb_feature_report.png") 
        plt.show()
        return 0
    
    def eval(self, x_valid, y_valid):
        y_pred = self.model.predict(x_valid)
        res = ("Evaluation Report (LightGBM)\n\n" +
                "RMSE: " + str(root_mean_squared_error(y_valid, y_pred)) + "\n" +
                "MAE: " + str(mean_absolute_error(y_valid, y_pred)) + "\n")
                # + "Accuracy (1-MAPE): " + str(mean_absolute_percentage_error(y_valid, y_pred)))
        helper.message(res)
        return 0

    def training_report(self):
        fig = lgb.plot_metric(self.train_loss)
        plt.savefig("graphs/lgb_training_report.png") 
        plt.show()
        return 0

    def eval(self, x_test, y_test):
        y_pred = self.model.predict(x_test)
        res = ("Evaluation Report (LightGBM)\n\n" +
                "RMSE: " + str(root_mean_squared_error(y_test, y_pred)) + "\n" +
                "MAE: " + str(mean_absolute_error(y_test, y_pred)) + "\n")
                # + "Accuracy (1-MAPE): " + str(mean_absolute_percentage_error(y_test, y_pred)))
        helper.message(res)
        return 0


class CatBoost:
    model = None

    def get_param(self, x_train, x_valid, y_train, y_valid):
        # Use optuna to find the HyperParameter
        helper.message("[INFO] Tuning hyperparameter (CatBoost) ...")
        param = catboost_search.getHyperParameter(x_train, x_valid, y_train, y_valid)

        # Add compulsory hyperparameter
        param["cat_features"] = ["make", "model", "trim", "body", "transmission", "color", "interior", "seller", "saledate_day", "saledate_month", "saledate_year"]
        param["verbose"] = 200
        param["thread_count"] = 8
        param["task_type"] = "CPU"
        param["iterations"] = 200
        
        helper.message("[INFO] Tuning completed.")
        helper.message(param)
        return param
       
    def train(self, x_train, y_train, x_valid, y_valid):
        # Tune hyperparameter
        param = self.get_param(x_train, x_valid, y_train, y_valid)

        # Create model
        helper.message("[INFO] Training model with tuned hyperparameter ... ")
        self.model = CatBoostRegressor(**param)
        self.model.fit(
            x_train,
            y_train,
            eval_set=(x_valid, y_valid),
            use_best_model=True
        )
        return 0

    def feature_report(self, x_train, y_train):
        # initialize JavaScript Visualization Library
        shap.initjs()
        helper.message("[INFO] Training explainer for CatBoost ...")
        tree = shap.TreeExplainer(self.model).shap_values(x_train)
        helper.message("[INFO] Training completed, visualising...")
        shap.summary_plot(tree, x_train, plot_type="bar",show=False)
        plt.savefig("graphs/cbt_feature_report.png") 
        plt.show()
        return 0

    def training_report(self):
        evals_result = self.model.get_evals_result()
        train_loss = evals_result["learn"]["RMSE"]
        test_loss = evals_result["validation"]["RMSE"]

        # Plot the training progress
        iterations = np.arange(1, len(train_loss) + 1)

        plt.figure(figsize=(7, 4))
        plt.plot(iterations, train_loss, label="Training Loss", color="blue")
        plt.plot(iterations, test_loss, label="Validation Loss", color="orange")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("CatBoost Training Progress")
        plt.legend()
        plt.grid()
        plt.savefig("graphs/cbt_training_report.png") 
        plt.show() 
        return 0

    def eval(self, x_test, y_test):
        y_pred = self.model.predict(x_test)
        res = (("Evaluation Report (CatBoost)\n\n" +
                "RMSE: " + str(root_mean_squared_error(y_test, y_pred)) + "\n" +
                "MAE: " + str(mean_absolute_error(y_test, y_pred)) + "\n"))
                # + "Accuracy (1-MAPE): " + str(mean_absolute_percentage_error(y_test, y_pred)))
        helper.message(res)
        return 0
