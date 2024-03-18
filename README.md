## Stock Market Prediction Web Application using Python Streamlit

Welcome to the Stock Market Prediction web application! This project leverages Python and Streamlit to provide users with insights into stock market trends and predictions.


## Overview

1. Enter Stock Ticker: Begin by entering the stock ticker symbol (e.g., "TSLA" for Tesla) to identify the stock of interest.

2. Data Description: Explore the comprehensive data description generated using the df.describe() function. This provides a snapshot of key statistics for the chosen stock.

3. Data Selection: We have curated data from 2010 to 2019, dividing it into training (70%) and testing (30%) sets. The final prediction is based on the testing set, ensuring robust evaluation.

4. Graphical Representations:
   
  a. Closing Price Prediction Chart: Visualize the predicted closing prices alongside the original values.
  
  b. 100 Days Moving Average: Plot the 100-day moving average, offering insights into trends by considering the previous 100 days' closing values.
  
  c. Time Chart vs 100-Day Moving Average and 200-Day Moving Average: Analyze the relationship between time and these moving averages, a technique often followed        by technical analysts for market predictions.

5. Predicted vs Actual Chart: Gain a comprehensive analysis of the data by comparing predicted values with actual results.

## Data Overview

Features: High, Low, Open, Close

Time Frame: Choose your desired time frame to observe predicted days.

## Prerequisites

- Python 3.00 version or above
- Libraries:
  ```
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  import pandas_datareader as data
  from alpha_vantage.timeseries import TimeSeries
  from sklearn.preprocessing import MinMaxScaler
  from keras.layers import Dense, Dropout, LSTM
  from keras.models import Sequential
  ```

## Usage

Using this command we can run the code in our command prompt

```python
python <file_name>.py
```

Way to run on streamlit, run the command in cmd. (make sure to run it where you have kept the python file)

```streamlit
streamlit run <file_name>.py
```

## Data Collection

Collected the stock market data and any external APIs or libraries used (e.g., alpha_vantage, pandas_datareader).

```h5
keras_model.h5
```
Another way to save the same data in keras model format

```keras
my_model.keras
```

## Machine Learning Algorithms

In this project, the primary machine learning algorithm employed is the Long Short-Term Memory (LSTM) model. The LSTM model is a type of recurrent neural network (RNN) designed to handle sequential and time-series data efficiently.

LSTM Model for Stock Market Prediction
Objective: The primary goal is to leverage historical data, specifically the past 100-day and 200-day moving averages, to make accurate predictions about future stock market trends.

Implementation:

Data Preparation: The dataset is organized into a suitable format, focusing on relevant features such as high, low, open, and close values.

Sequence Formation: A sequential dataset is created, capturing the temporal dependencies of the stock market data. For this purpose, the LSTM model considers the past 100 and 200 days.

Model Architecture: The LSTM architecture is employed due to its ability to capture long-term dependencies in sequential data. The model is trained to learn patterns from historical stock market trends.

Training and Testing: The dataset is divided into training (70%) and testing (30%) sets. The model is trained on the historical data, and its performance is evaluated on the testing set to ensure generalization.

Visualization - Predicted vs Actual

The final step involves generating a chart that visually compares the predicted values from the LSTM model with the actual values. This comparison allows for a comprehensive analysis of the model's accuracy and effectiveness in forecasting stock market trends.

## Challenges faced and what I learned from this project?

1. Python Libraries and Their Usages

Challenge: Working with a variety of Python libraries such as NumPy, Pandas, Matplotlib, and others posed initial challenges in understanding their functionalities and integration into the project.

What I Learned: Through overcoming these challenges, I gained a deeper understanding of efficient data manipulation, visualization, and analysis using Python libraries. This experience enhanced my proficiency in leveraging diverse tools within the Python ecosystem.

2. Utilizing API for Data Retrieval

Challenge: Incorporating external data using an API, in this case from another website, introduced complexities related to data retrieval, handling, and ensuring consistent updates.

What I Learned: Overcoming API-related challenges expanded my knowledge of data fetching and integration. I learned to implement robust solutions for seamless data extraction, fostering skills in API utilization and data synchronization.

3. Implementing ML LSTM Model

Challenge: Developing and fine-tuning a Machine Learning Long Short-Term Memory (LSTM) model for stock market prediction presented challenges in model architecture, hyperparameter tuning, and ensuring accurate predictions.

What I Learned: The process of implementing an ML LSTM model enhanced my understanding of time series forecasting and the intricacies of LSTM networks. This experience provided insights into optimizing model performance and effectively handling sequential data.
