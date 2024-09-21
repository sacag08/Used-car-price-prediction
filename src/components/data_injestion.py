import os
import pandas as pd
import numpy as np
import pickle
import sys
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTraining
from src.components.hyper_parameter_tuner import HyperparameterTuning
from src.components.data_predictor import Prediction
from src.pipeline.predict_pipeline import PredictPipeline,CustomData

from sklearn.model_selection import train_test_split
@dataclass
class DataInjestionConfig:
    train_data_path : str = os.path.join('artifacts','train.csv')
    test_data_path : str = os.path.join('artifacts','test.csv')
    raw_data_path : str = os.path.join('artifacts','raw.csv')


class DataInjestion:
    def __init__(self):
        self.DataInjestionConfig = DataInjestionConfig()
    
    def initiate_data_injestion(self):
        logging.info('Initiated Data Injestion')
        try:
            data = pd.read_csv('C:/Projects/Used Car Price prediction/src/Notebook/data/train.csv')
            raw_data = pd.read_csv('C:/Projects/Used Car Price prediction/src/Notebook/data/test.csv')
            os.makedirs(os.path.dirname(self.DataInjestionConfig.train_data_path),exist_ok=True)
            data.to_csv(DataInjestionConfig.train_data_path)
            logging.info('Read and wrote the data to artifacts')
            train_data,test_data = train_test_split(data,test_size=0.3,random_state=41)
            train_data.to_csv(DataInjestionConfig.train_data_path)
            test_data.to_csv(DataInjestionConfig.test_data_path)
            raw_data.to_csv(DataInjestionConfig.raw_data_path)
            logging.info('Splitted the data in test and train datasets and pasted to artifacts')
            return (
                self.DataInjestionConfig.train_data_path,
                self.DataInjestionConfig.test_data_path,
                self.DataInjestionConfig.raw_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__ =="__main__":
    obj = DataInjestion()
    train_path,test_path,raw_data_path = obj.initiate_data_injestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,obj = data_transformation.initiate_data_transformation(train_path=train_path,test_path=test_path)
    # modeltraining = ModelTraining(train_data=train_arr,test_data=test_arr)
    # modeltraining.initiate_model_training()
    # xtrain = train_arr[:,:-1]
    # ytrain = train_arr[:,-1
    # xtest = test_arr[:,:-1]
    # ytest = test_arr[:,-1]
    # tuning = HyperparameterTuning(xtrain=xtrain,xtest=xtest,ytest=ytest,ytrain=ytrain)
    # tuning.initiate_hyperparameter_tuner(['LightGBM'])

    # transformed_raw_data = data_transformation.transform_raw_data_from_path(raw_data_path=raw_data_path,preprocessor_path = obj)
    # predict = Prediction(raw_data=transformed_raw_data,train_data=train_arr,raw_path=raw_data_path)
    # predict.predict_pipeline()
    # data = CustomData(id='100', brand="Cherokee", model ="2021 Jeep Grand Cherokee", model_year = 2024, milage = 100250, fuel_type = "Gasoline", engine = "Pentastar 3.6L V6", transmission = "A/T", ext_col=  "Velvet Red Pearl Coat", int_col = "black", accident = "None Reported", clean_title = "Yes")
    # d = data.get_data_as_dataframe()
    # pred = PredictPipeline()
    # print(pred.predict(data=d))