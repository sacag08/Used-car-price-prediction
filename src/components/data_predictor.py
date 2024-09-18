import os
import sys 
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingRegressor

from src.utils import load_object,save_object
from src.logger import logging
from src.exception import CustomException

from dataclasses import dataclass
from src.components.data_transformation import DataTransformation

@dataclass
class Data_Predictor_config:
    Data_Predictor_file_path_xgb = os.path.join('artifacts','xgb_prediction.csv')
    Data_Predictor_file_path_lgbm = os.path.join('artifacts','lgbm_prediction.csv')
    Data_Predictor_file_path_vr= os.path.join('artifacts','vr_prediction.csv')
    Data_Preprocessor_path = os.path.join('artifacts','preprocessing.pkl')
    xgb_model_path = os.path.join('artifacts','xgb_tuned.pkl')
    lbg_model_path = os.path.join('artifacts','lgbm_tuned.pkl')

class Prediction:
    def __init__(self,raw_data,train_data,raw_path):
        self.Data_Predictor_file_path = Data_Predictor_config()
        self.raw_data = raw_data
        self.train_data = train_data
        self.raw_path = raw_path

    def predict_pipeline(self):
        try:
            logging.info('Predict pipeline started')
            xtrain = self.train_data[:,:-1]
            ytrain = self.train_data[:,-1]

            raw_id = pd.read_csv(self.raw_path)['id']
            
            logging.info('Raw data transformed')
            xgb_model = load_object(file_path=self.Data_Predictor_file_path.xgb_model_path)
            lgb_model = load_object(file_path=self.Data_Predictor_file_path.lbg_model_path)
            logging.info('models loaded')
            vr = VotingRegressor(estimators=[
                ('XGB',xgb_model),
                ('lgbm',lgb_model)]
            )
            vr.fit(xtrain,ytrain)
            logging.info('Voting regressor trained')
            predictions = vr.predict(self.raw_data)
            logging.info('Predictioned completed')
            submission_df = pd.DataFrame({
                'id': pd.Series(raw_id),
                'class': pd.Series(predictions)})

            submission_df.to_csv(self.Data_Predictor_file_path.Data_Predictor_file_path_vr, index=False)
            print("Submission file created: submission_xgb1.csv")
        except Exception as e:
            raise CustomException(e,sys)

        

    def initiate_prediction(self,transformed_raw_data):
        pass
        
