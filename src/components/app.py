import streamlit as st
import pandas as pd 

from src.logger import logging
from src.exception import CustomException
from src.utils import load_object

from src.pipeline.predict_pipeline import PredictPipeline,CustomData

features = load_object('artifacts/features.pkl')
st.title(" Welcome to Used Cars Price prediction model")
st.write("Input Features")

id : int = 1
brand :str = st.selectbox("Enter the car brand of the vehicle",options=features['cleaned_brand'])
model :str = st.selectbox("Enter the model of the vehicle",options=features['cleaned_model'])
model_year :int = st.slider("Enter model year of the vehicle",min_value=1990,max_value=2024)
milage :int = st.slider("Enter current millage of the vehicle",min_value=0,max_value = 99999)
fuel_type :str = st.selectbox("Enter the fuel type for the vehicle",options=features['fuel_type'])
engine: str = st.text_input("Enter the engine type for the vehicle")
transmission: str = st.selectbox("Enter the transmission type for the vehicle",options=features['transmission_type'])
ext_col : str = st.selectbox("Enter the exterior color the vehicle",options=features['ext_col'])
int_col :str = st.selectbox("Enter the  interior color the vehicle",options=features['int_col'])
accident : str =  st.selectbox("Enter Yes/No if the vehicle had reported any accidents ",options=['Yes','No'])
clean_title : str = st.text_input("Enter Yes/No if the vehicle has a clean title",options=['Yes','No'])

if accident == 'Yes':
    accident = "At least 1 accident or damage reported"
else:
    accident = "None reported"

data = CustomData(id=id, brand=brand, model =model, model_year = model_year, milage = milage, fuel_type = fuel_type, engine = engine, transmission = transmission, ext_col=  ext_col, int_col = int_col, accident = accident, clean_title = clean_title)
d = data.get_data_as_dataframe()
pred = PredictPipeline()
st.write(f'The price pridect is {pred.predict(data=d)}')