import os
import pandas as pd
import numpy as np
import pickle
import sys
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException

from src.components.data_transformation import DataTransformation

from sklearn.model_selection import train_test_split
@dataclass
class DataInjestionConfig:
    train_data_path : str = os.path.join('artifacts','train.csv')
    test_data_path : str = os.path.join('artifacts','test.csv')
    val_data_path : str = os.path.join('artifacts','val.csv')


class DataInjestion:
    def __init__(self):
        self.DataInjestionConfig = DataInjestionConfig()
    
    def initiate_data_injestion(self):
        logging.info('Initiated Data Injestion')
        try:
            data = pd.read_csv('C:/Projects/Used Car Price prediction/src/Notebook/data/train.csv')
            os.makedirs(os.path.dirname(self.DataInjestionConfig.train_data_path),exist_ok=True)
            data.to_csv(DataInjestionConfig.train_data_path)
            logging.info('Read and wrote the data to artifacts')
            train_data,test_data = train_test_split(data,test_size=0.3,random_state=41)
            train_data.to_csv(DataInjestionConfig.train_data_path)
            test_data.to_csv(DataInjestionConfig.test_data_path)
            logging.info('Splitted the data in test and train datasets and pasted to artifacts')
            return (
                self.DataInjestionConfig.train_data_path,
                self.DataInjestionConfig.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__ =="__main__":
    obj = DataInjestion()
    train_path,test_path = obj.initiate_data_injestion()
    data_transformation = DataTransformation()
    print(data_transformation.initiate_data_transformation(train_path=train_path,test_path=test_path))

