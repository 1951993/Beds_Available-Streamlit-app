import streamlit as st
import pandas as pd
import numpy as np
from fbprophet import Prophet
from fbprophet.diagnostics import performance_metrics
from fbprophet.diagnostics import cross_validation
from fbprophet.plot import plot_cross_validation_metric
import base64

st.title('Forecasting of Beds Available')

st.write("Note:follow the following instructions to upload the csv file")
st.write("Import the time series csv file. It should have two columns labelled as 'ds' and 'y'.The 'ds' column should be of datetime format  by Pandas. The 'y' column must be numeric representing the measurement to be forecasted.")

data = st.file_uploader('Upload here',type='csv')

df_prophet = pd.read_csv('C:/Users/fast/Desktop/Project-P65/Project P65/data.csv')
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'],errors='coerce') 
    
st.write(data)
    
max_date = df_prophet['ds'].max()

st.write("SELECT FORECAST PERIOD")

periods_input = st.number_input('How many days forecast do you want?',
min_value = 1, max_value = 365)


prophet_model = Prophet(interval_width=0.95, daily_seasonality=True)
prophet_model.fit(df_prophet)


st.write("VISUALIZE FORECASTED DATA")
st.write("The following plot shows future predicted values. 'yhat' is the predicted value; upper and lower limits are 80% confidence intervals by default")

future = prophet_model.make_future_dataframe(periods=periods_input)
    
fcst = prophet_model.predict(future)
forecast = fcst[['ds', 'yhat']]

forecast_filtered =  forecast[forecast['ds'] > max_date]    
st.write(forecast_filtered)
