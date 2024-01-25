import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
from keras.models import load_model
import streamlit as st

start = '2010-01-01'
end = '2023-12-31'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
ts = TimeSeries(key='E9EYV24M353PT2TL', output_format='pandas')
df, meta_data = ts.get_daily(symbol=user_input, outputsize='full')  
# 'full' retrieves all available historical data
df = df[(df.index >= start) & (df.index <= end)]

# Describing the data
st.subheader('Data from 2010 - 2023')
st.write(df.describe())

#Visualizations
# Closing Price VS Time chart
st.subheader('Closing Price VS Time chart')
fig1 = plt.figure(figsize=(12, 6))
plt.plot(df['4. close'])
st.pyplot(fig1)

# Closing Price vs Time Chart with 100MA
st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df['4. close'].rolling(100).mean()
fig2 = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(df['4. close'])
st.pyplot(fig2)

# Closing Price vs Time Chart with 100MA and 200MA
st.subheader('Closing Price vs Time Chart with 100MA and 200MA')
ma200 = df['4. close'].rolling(200).mean()
fig3 = plt.figure(figsize=(12, 6))
plt.plot(ma100, label='100MA')
plt.plot(ma200, label='200MA')
plt.plot(df['4. close'], label='Closing Price')
plt.legend()
st.pyplot(fig3)

#spliting data into training and testing

data_training = pd.DataFrame(df['4. close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['4. close'][int(len(df)*0.70): int(len(df))])

print(data_training.shape)
print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

#load the model
model = load_model(r'C:\Users\suraj\Stock_predction\keras_model.h5')

#testing part

past_100_days = data_training.tail(100)
final_df = past_100_days._append(data_testing, ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i,0])

x_test , y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Final Graph
st.subheader('Prediction vs Orginal')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Orginal Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)