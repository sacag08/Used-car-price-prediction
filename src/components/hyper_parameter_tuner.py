import pandas as pd
import numpy as np
import os
import sys
from src.logger import logging
from src.exception import CustomException

from src.utils import save_object
from sklearn.model_selection import train_test_split,KFold
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor,VotingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error,r2_score
import sklearn
import xgboost as xgb
import lightgbm as lgb
import optuna

from dataclasses import dataclass

@dataclass
class HyperparameterTuningConfig:
    tuned_xgboost_model_path = os.path.join('artifacts','xgb_tuned.pkl')
    tuned_lgbm_model_path = os.path.join('artifacts','lgbm_tuned.pkl')
    tuned_catboost_model_path = os.path.join('artifacts','catboost_tuned.pkl')
    tuned_randomforest_model_path = os.path.join('artifacts','random_forest_tuned.pkl')

class HyperparameterTuning:
    def __init__(self,xtrain,xtest,ytrain,ytest):
        self.HyperparameterTuningConfig  = HyperparameterTuningConfig()
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.ytest = ytest
    
    def xgb_model_tuner(self,trial):
    
        dtrain = xgb.DMatrix(self.xtrain, label=self.ytrain)
        dtest = xgb.DMatrix(self.xtest, label=self.ytest)
        logging.info('Dmatrix created for XGBoost HyperParameter tuning')

        param = {
            "silent": 1,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
            "lambda": trial.suggest_loguniform("lambda", 1e-5, 1.0),
            "alpha": trial.suggest_loguniform("alpha", 1e-5, 1.0),
        }

        if param["booster"] == "gbtree" or param["booster"] == "dart":
            param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
            param["eta"] = trial.suggest_loguniform("eta", 1e-5, 1.0)
            param["gamma"] = trial.suggest_loguniform("gamma", 1e-5, 1.0)
            param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            param["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-5, 1.0)
            param["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-5, 1.0)

        # Add a callback for pruning.
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-rmse")
        bst = xgb.train(param, dtrain, evals=[(dtest, "validation")], callbacks=[pruning_callback])
        preds = bst.predict(dtest)
        mse = sklearn.metrics.mean_squared_error(self.ytest, preds)
        return mse
        
        
    
    def tune_lgbm_model(self,trial):
            params = {
            'n_estimators': trial.suggest_int('n_estimators', 800, 1500),
            'num_leaves': trial.suggest_int('num_leaves', 75, 200),
            'max_depth': trial.suggest_int('max_depth', 12, 30),
            'cat_smooth': trial.suggest_int('cat_smooth', 20, 120),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.02),
            'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.4, 0.8),
            'min_split_gain': trial.suggest_loguniform('min_split_gain', 1e-4, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 70),
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-10, 1e-3),
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-10, 1e-2),
            #'scale_pos_weight': trial.suggest_uniform('scale_pos_weight', 1.0, 12.0),
            'max_bin': trial.suggest_int('max_bin', 200, 1000),
            }

            # Create LightGBM datasets
            dtrain = lgb.Dataset(self.xtrain, label=self.ytrain)
            dvalid = lgb.Dataset(self.xtest, label=self.ytest, reference=dtrain)

            # Train the model
            model = lgb.train(params, dtrain, valid_sets=[dvalid], 
                            callbacks=[lgb.early_stopping(stopping_rounds=3), lgb.log_evaluation(50)])

            # Predict on the validation set
            y_pred_valid = model.predict(self.xtest)

            # Calculate RMSE on the validation set
            rmse = mean_squared_error(self.ytest, y_pred_valid, squared=False)

            return rmse


    
    def initiate_hyperparameter_tuner(self,models):
        try:
            for model in models:
                if model == 'XGBRegressor':
                    logging.info('XGBoost HyperParameter tuning started')
                    study = optuna.create_study(direction='minimize')
                    study.optimize(self.xgb_model_tuner, n_trials=50)
                    best_params = study.best_params
                    obj=XGBRegressor(**best_params).fit(self.xtrain,self.ytrain)
                    save_object(obj=obj,file_path=self.HyperparameterTuningConfig.tuned_xgboost_model_path)
                    return best_params
                if model == 'LightGBM':
                    logging.info('LightGBM hyperparameter tuning started ')
                    study = optuna.create_study(direction='minimize')
                    study.optimize(self.tune_lgbm_model,n_trials=50)
                    best_params = study.best_params
                    obj=LGBMRegressor(**best_params).fit(self.xtrain,self.ytrain)
                    save_object(obj=obj,file_path=self.HyperparameterTuningConfig.tuned_lgbm_model_path)
                    return best_params

        except Exception as e:
            raise CustomException(e,sys)
