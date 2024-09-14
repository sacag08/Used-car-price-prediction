import pandas as pd
import numpy as np
import os
import sys
import re
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.utils import fill_fuel_type,save_object


from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler,MinMaxScaler,MaxAbsScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
@dataclass
class DataTransfromationConfig:
    preprocessing_file_path: str=os.path.join('artifacts','preprocessing.pkl')

class DataTransformation:
    def __init__(self):
        self.DataTransfromationConfig = DataTransfromationConfig()
    
    def initiate_data_extraction(self,df):
        try:
            df['transmission'] = df['transmission'].str.lower()
            df['transmission_type'] = df['transmission'].apply(lambda x:
                'manual' if 'm/t' in x or 'manual' in x or 'mt' in x else 
                'automatic' if 'a/t' in x or 'automatic' in x else
                'CVT' if 'CVT' in x else
                'dual' if 'dual' in x else 'other'
            )
            df['hoursepower'] = df['engine'].str.extract(r'(\d+\.\d+)(?=HP)').astype(float)
            df['capacity'] = df['engine'].str.extract(r'(\d+\.\d+)(?=L| Liter)').astype(float)
            df['Cylinder'] = df['engine'].apply(lambda x: x if pd.isnull(x)
                                            else float(re.search('(\d)\s(Cylinder)',x).group(1)) if re.search('(\d)\s(Cylinder)',x)
                                            else float(re.search('\s(V)(\d)', x ).group(2)) if re.search('\s(V)(\d)', x) else np.nan)
            return df
        
        except Exception as e:
            raise CustomException(e,sys)
            
    def fill_missing_value(self,df):
        try:

            column = df.columns
            numeric_features = [feature for feature in df.columns if df[feature].dtype != 'O']
            categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']
            most_common = dict()
            for i in categorical_features:
                value = str(df[i].value_counts()[:1].index.values[0])
                most_common[i] = value

            df['fuel_type'] = df[['engine','fuel_type']].apply(lambda x : fill_fuel_type(x),axis=1)
            df['fuel_type'].fillna('Gasoline',inplace=True)
            df['accident'].fillna('None reported',inplace=True)
            df['clean_title'].fillna('Yes' if 'None reported' in df['accident'] else 'No',inplace=True)
            for i in categorical_features:
                df[i] = df[i].replace('–',most_common[i])
            common_color = ['black', 'white', 'gray', 'silver', 'brown', 'red', 'blue', 'green',
                'beige', 'tan', 'orange', 'gold', 'yellow', 'purple', 'pink', 
                'charcoal', 'ivory', 'camel', 'chestnut', 'pearl', 'linen', 'graphite',
                'copper', 'slate', 'bronze', 'sand', 'amber','macchiato','ebony','cocoa']
            
            df['int_col'] = df['int_col'].apply(lambda x: x if [color for color in common_color if color in str.lower(x).split(' ')] == [] else [color for color in common_color if color in str.lower(x).split(' ')][0])
            df['ext_col'] = df['ext_col'].apply(lambda x: x if [color for color in common_color if color in str.lower(x).split(' ')] == [] else [color for color in common_color if color in str.lower(x).split(' ')][0])


            df['interior_rare_color'] = df['int_col'].apply(lambda x: 1 if str.lower(x) not in common_color else 0)
            df['exterior_rare_color'] = df['ext_col'].apply(lambda x: 1 if str.lower(x) not in common_color else 0)
            
            luxury_brands = ["Mercedes-Benz", "BMW", "Audi", "Porsche", "Land Rover","Land"
            "Lexus", "Cadillac", "Tesla", "INFINITI", "Jaguar", 
            "Bentley", "Maserati", "Lamborghini", "Genesis", "Rolls-Royce", 
            "Ferrari", "McLaren", "Aston Martin", "Lucid", "Lotus", 
            "Karma", "Bugatti", "Maybach"]

            df['is_luxry_brand'] = df['brand'].apply(lambda x: 1 if x in luxury_brands else 0)

            df['age'] = df['model_year'].apply(lambda x: int(2025 - x))

            df['mile/year'] = df['milage']/df['age']


            model_sample = df['model'].value_counts()
            low_models_samples = list(model_sample[model_sample.values < 101].index)
            df['cleaned_model'] = df['model'].apply(lambda x: x if x not in low_models_samples else 'others')

            df.drop(['id','brand','engine','model_year','transmission'],axis=1,inplace=True)
            return df
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def get_data_transformer_object(self):
        try:
            numeric_features = ['milage','hoursepower', 'capacity','Cylinder', 'interior_rare_color', 'exterior_rare_color','is_luxry_brand', 'age', 'mile/year']
            categorical_features = ['cleaned_model','fuel_type','transmission_type','accident','clean_title','int_col','ext_col','model']
            numeric_features_pipeline = Pipeline(
                [   
                    ('fillna',SimpleImputer(strategy='median')),
                    ('Scale',MaxAbsScaler())
                ]
            )

            categorical_features_pipeline = Pipeline(
                [
                    ('ohe',OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=True)),
                    ('sclae',MaxAbsScaler())
                ]
            )

            transformer = ColumnTransformer(
                [
                    ('numerical',numeric_features_pipeline,numeric_features),
                    ('categorical',categorical_features_pipeline,categorical_features)
                ]
            )
            return transformer
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info('Data Transformation begins')
            target_feature = 'price'
            train_dataframe = pd.read_csv(train_path)
            test_dataframe = pd.read_csv(test_path)
            logging.info('Training and Test data read')

            train_data = train_dataframe.drop(target_feature,axis=1)
            test_data = test_dataframe.drop(target_feature,axis=1)

            train_target_data = train_dataframe[target_feature]
            test_target_data = test_dataframe[target_feature]
            logging.info('Target and feature data splited')

            extracted_train_data = self.initiate_data_extraction(train_data)
            extracted_test_data = self.initiate_data_extraction(test_data)
            logging.info('Data Extraction completed')

            filled_train_data = self.fill_missing_value(extracted_train_data)
            filled_test_data = self.fill_missing_value(extracted_test_data)
            logging.info('Data filling completed')

            preprocessor = self.get_data_transformer_object()
            logging.info('Preprocessor object returned')
            transformed_train_Data = preprocessor.fit_transform(filled_train_data).toarray()
            transformed_test_Data = preprocessor.transform(filled_test_data).toarray()
            logging.info('Train and test data transformed')

            train_arr = np.c_[
                transformed_train_Data, np.array(train_target_data)
            ]
            logging.info('train data converted to arrays')
            test_arr = np.c_[
                transformed_test_Data, np.array(test_target_data)
            ]
            logging.info('Train and test data converted to arrays')
            save_object(
                file_path= self.DataTransfromationConfig.preprocessing_file_path,
                obj = preprocessor
                )
            logging.info('Preprocessor object saved')
            return (train_arr,test_arr,self.DataTransfromationConfig.preprocessing_file_path)

        except Exception as e:
            raise CustomException(e,sys)
