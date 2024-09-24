import streamlit as st
import pandas as pd 

from src.logger import logging
from src.exception import CustomException


from src.pipeline.predict_pipeline import PredictPipeline,CustomData

st.title(" Welcome to Used Cars Price prediction model")
st.write("Input Features")

id : int = st.number_input("Enter Vehicle ID")
brand :str = st.text_input("Enter the car brand of the vehicle")
model :str = st.text_input("Enter the model of the vehicle")
model_year :int = st.number_input("Enter model year of the vehicle")
milage :int = st.number_input("Enter current millage of the vehicle")
fuel_type :str = st.text_input("Enter the fuel type for the vehicle")
engine: str = st.text_input("Enter the engine type for the vehicle")
transmission: str = st.text_input("Enter the transmission type for the vehicle")
ext_col : str = st.text_input("Enter the exterior color the vehicle")
int_col :str = st.text_input("Enter the  interior color the vehicle")
accident : str =  st.text_input("Enter Yes/No if the vehicle had reported any accidents ")
clean_title : str = st.text_input("Enter Yes/No if the vehicle has a clean title")

if accident == 'Yes':
    accident = "At least 1 accident or damage reported"
else:
    accident = "None reported"

data = CustomData(id=id, brand=brand, model =model, model_year = model_year, milage = milage, fuel_type = fuel_type, engine = engine, transmission = transmission, ext_col=  ext_col, int_col = int_col, accident = accident, clean_title = clean_title)
d = data.get_data_as_dataframe()
pred = PredictPipeline()
st.write(f'The price pridect is {pred.predict(data=d)}')