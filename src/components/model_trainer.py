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
class ModelTrainingConfig:
    trained_model_path = os.path.join('artifacts','model.pkl')
    score_matrix = os.path.join('artifacts','score.csv')

class ModelTraining:
    def __init__(self,train_data,test_data):
        self.ModelTrainingConfig = ModelTrainingConfig()
        self.train_data = train_data
        self.test_data = test_data

    def base_model_scoring(self,models,xtrain,ytrain,xtest,ytest):
        try:
            train_r2_score = []
            test_r2_score = []
            model_name = []
            trained_models = []
            train_mse = []
            test_mse = []
            for model in list(models):
                trained_model = models[model].fit(xtrain,ytrain)
                trained_models.append(trained_model)
                print('model trained for {}'.format(model))
                model_name.append(model)
                train_model_prediction = trained_model.predict(xtrain)
                val_model_prediction = trained_model.predict(xtest)
                model_train_r2_score,model_train_mean_squared_error = r2_score(ytrain,train_model_prediction),mean_squared_error(ytrain,train_model_prediction)
                model_test_r2_score,model_test_mean_squared_error = r2_score(ytest,val_model_prediction),mean_squared_error(ytest,val_model_prediction)
                test_r2_score.append(model_test_r2_score)
                train_r2_score.append(model_train_r2_score)
                train_mse.append(model_train_mean_squared_error)
                test_mse.append(model_test_mean_squared_error)
                print("Train and test Scores for {} is {} and {}. The MSE for train and test data is {} and {}".format(model,train_r2_score,test_r2_score,model_train_mean_squared_error,model_test_mean_squared_error))
                scores = pd.DataFrame({'model': model_name,'train_r2_score':train_r2_score,'train_mse':train_mse,'test_score':test_r2_score,'test_mse':test_mse})
            
            return scores

        except Exception as e:
            raise CustomException(e,sys)

    
    def initiate_model_training(self):
        try:
            xtrain = self.train_data[:,:-1]
            ytrain = self.train_data[:,-1]

            xtest = self.test_data[:,:-1]
            ytest = self.test_data[:,-1]
            logging.info('Read train and test data in module model trainer')
            models = {
                'linear_regression':LinearRegression(),
                # 'lasso_regression' :Lasso(),
                # 'Ridge_regression' :Ridge(),
                'adaboost_regressor' : AdaBoostRegressor(
                    DecisionTreeRegressor( max_depth=2,
                                                                random_state=0,
                                                                ),
                    n_estimators=30),
                'RandomForestRegressor': RandomForestRegressor(max_depth =4,min_samples_split=50000,max_samples=0.7,max_features=0.9,n_estimators=20),
                'XGBRegressor' : XGBRegressor(max_depth = 5, subsample = 0.7,min_samples_split=50000),
                'CatBoostRegressor' : CatBoostRegressor(depth =5),
                'LightGBM' : LGBMRegressor()
            }

            logging.info('models dict created')
            scores = self.base_model_scoring(models=models,xtrain=xtrain,xtest=xtest,ytrain=ytrain,ytest=ytest)
            scores.to_csv(self.ModelTrainingConfig.score_matrix)
            logging.info('score file creted')
        except Exception as e:
            raise CustomException(e,sys)


        
        
        
            
