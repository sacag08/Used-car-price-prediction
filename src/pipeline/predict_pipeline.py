import pandas as pd 
from src.utils import load_object,fill_missing_value,initiate_data_extraction
from src.components.data_transformation import DataTransformation
from src.logger import logging
from src.exception import CustomException
import os
import sys

class PredictPipeline():
    def __init__(self):
        pass
    def predict(self,data):
        try:
            model = load_object(file_path=os.path.join("artifacts","lgbm_tuned.pkl"))
            preprocesor = load_object(file_path=os.path.join("artifacts","preprocessing.pkl"))
            transformed_data = preprocesor.transform(fill_missing_value(initiate_data_extraction(data)))
            pred = model.predict(transformed_data)
            return pred
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
            id : int,
            brand :str,
            model :str,
            model_year :int,
            milage :int,
            fuel_type :str,
            engine: str,
            transmission: str,
            ext_col : str,
            int_col :str,
            accident : str,
            clean_title : str):
        self.id=id
        self.brand=brand
        self.model=model
        self.model_year=model_year
        self.milage=milage
        self.fuel_type=fuel_type
        self.engine=engine
        self.transmission= transmission
        self.ext_col= ext_col
        self.int_col= int_col
        self.accident = accident
        self.clean_title = clean_title
    
    def get_data_as_dataframe(self):
        try:
            custom_data_dict = {
                'id': [self.id],
                'brand': [self.brand],
                'model': [self.model],
                'model_year': [self.model_year],
                'milage' : [self.milage],
                'fuel_type': [self.fuel_type],
                'engine': [self.engine],
                'transmission': [self.transmission],
                'ext_col': [self.ext_col],
                'int_col': [self.int_col],
                'accident': [self.accident],
                'clean_title': [self.clean_title]
            }
            return pd.DataFrame(custom_data_dict)
        except Exception as e:
            raise CustomException(e,sys)