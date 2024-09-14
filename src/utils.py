import os
import sys
import pandas as pd
import numpy as np  
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from src.exception import CustomException
import dill
from src.logger import logging    


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

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
