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
        try:   
            dtrain = xgb.DMatrix(self.xtrain, label=self.ytrain)
            dtest = xgb.DMatrix(self.xtest, label=self.ytest)
            logging.info('Dmatrix created for XGBoost HyperParameter tuning')

            param = {
                "silent": 1,
                "objective": trial.suggest_categorical("objective",["reg:squarederror","reg:linear"]),
                "eval_metric": trial.suggest_categorical("eval_metric", ["rmse", "logloss"]),
                "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
                "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
                "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
            }

            if param["booster"] == "gbtree" or param["booster"] == "dart":
                param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
                param["eta"] = trial.suggest_loguniform("eta", 1e-8, 1.0)
                param["gamma"] = trial.suggest_loguniform("gamma", 1e-8, 1.0)
                param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
            if param["booster"] == "dart":
                param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
                param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
                param["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-8, 1.0)
                param["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)

            # Add a callback for pruning.
            pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-rmse")
            bst = xgb.train(param, dtrain, evals=[(dtest, "validation")], callbacks=[pruning_callback])
            preds = bst.predict(dtest)
            mse = sklearn.metrics.mean_squared_error(self.ytest, preds)
            return mse
        except Exception as e:
            raise CustomException(e,sys)

    
    def initiate_hyperparameter_tuner(self,models):
        try:
            for model in models:
                if model == 'XGBRegressor':
                    logging.info('XGBoost HyperParameter tuning started')
                    study = optuna.create_study(direction='minimize')
                    study.optimize(self.xgb_model_tuner, n_trials=100)
                    best_params = study.best_params
                    return best_params

        except Exception as e:
            raise CustomException(e,sys)
