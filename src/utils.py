import os
import sys
import pandas as pd
import numpy as np  
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from src.exception import CustomException
import dill
from src.logger import logging    
import re


def fill_fuel_type(x):
        try:
            if pd.isnull(x[1]):
                if 'gasoline' in str.lower(x[0]):
                    return 'Gasoline'
                elif 'flex' in str.lower(x[0]):
                    return 'E85 Flex Fuel'
                elif (('plug-in'in str.lower(x[0])) or ('electric/gas' in str.lower(x[0]))):
                    return 'Plug-In Hybrid'
                elif 'hybrid' in str.lower(x[0]):
                    return 'Hybrid'
                elif 'electric' in str.lower(x[0]):
                    return 'electric'
                else:
                    return np.nan 
            else:
                return x[1]
        except Exception as e:
            raise CustomException(e,sys)

def initiate_data_extraction(df):
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

def fill_missing_value(df):
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

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)