# Hunter Johnson

#!/usr/bin/env python
# coding: utf-8

# In[44]:


import pandas as pd
import yfinance as yf
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
from sklearn.metrics import mean_squared_error
train_start = '2010-01-01' # Training start date
train_end = '2018-12-31' # Training and date
forecast_start = '2019-01-01' # Forecast start date
forecast_end = '2019-12-31' # Forecast end date
stock_symbols = ['AMZN', 'MSFT'] # Define stock symbols of interest
data = yf.download(stock_symbols, start=train_start, end=forecast_end, interval='1d')['Adj Close'] # Use yahoo finance to pull stock data
data.index = pd.to_datetime(data.index) # Set data index
data = data.asfreq('B')  # Define data frequency as daily
data = data.ffill() # fill missing data 
train_data = data[train_start:train_end] # Create training data frame
forecast_dates = pd.date_range(start=forecast_start, end=forecast_end, freq='B') # Create test frame
forecast_actual_data = data[forecast_start:forecast_end] #Create test frame acual
train_data = train_data.dropna() # drop NA's
forecast_actual_data = forecast_actual_data.dropna() # drops NA's
train_data_diff = train_data.diff().dropna() # Make data stationary
# Plot the training data
plt.figure(figsize=(10, 6))
plt.plot(train_data.index, train_data['AMZN'], label='AMZN Training Data')
plt.plot(train_data.index, train_data['MSFT'], label='MSFT Training Data')
plt.title("Training Stock Prices (2010-2018)")
plt.legend()
plt.show()
tscv = TimeSeriesSplit(n_splits=5) # Set up 5-fold cross validaition
def cross_validate_VAR(data, maxlags): # set up VAR CV function 
    errors = [] # intialize error array
    for train_index, test_index in tscv.split(data): # create folds
        train, test = data.iloc[train_index], data.iloc[test_index] # establish training and test data
        model = VAR(train) # create training model
        results = model.fit(maxlags=maxlags) #gather result for given lag amount
        forecast = results.forecast(train.values[-results.k_ar:], steps=len(test)) #create forecast with current model
        error = np.mean((forecast - test.values) ** 2) #calculate MSE
        errors.append(error) # append error array
    return np.mean(errors) # return mean of errors
lag_lengths = range(1, 16) # create range of possible lags
errors = [cross_validate_VAR(train_data_diff, lag) for lag in lag_lengths] # append errors for each lag length
optimal_lag = lag_lengths[np.argmin(errors)] # find min mse and optimal lag
# Plot the cross-validation errors
plt.figure(figsize=(10, 6))
plt.plot(lag_lengths, errors, marker='o')
plt.title('Cross-Validation Errors for Different Lag Lengths')
plt.xlabel('Lag Length')
plt.ylabel('Mean Squared Error')
plt.show()
model = VAR(train_data_diff) #create optimal VAR model
results = model.fit(optimal_lag)  # Pull results of VAR model
print(results.summary()) # Print results
forecast_diff = results.forecast(train_data_diff.values[-results.k_ar:], steps=len(forecast_dates)) # Forecast 2019 using optimal model 
last_values = train_data.values[-1] #convert forecasted values
forecast = last_values + np.cumsum(forecast_diff, axis=0) # update forecast values
forecast_df = pd.DataFrame(forecast, index=forecast_dates, columns=train_data.columns) #Create forecast data frame
# Plot the forecasted data
plt.figure(figsize=(10, 6))
plt.plot(forecast_df.index, forecast_df['AMZN'], label='Forecasted AMZN', linestyle='--', color='blue')
plt.plot(forecast_df.index, forecast_df['MSFT'], label='Forecasted MSFT', linestyle='--', color='orange')
# Plot the actual data for 2019
plt.plot(forecast_actual_data.index, forecast_actual_data['AMZN'], label='Actual AMZN', color='blue')
plt.plot(forecast_actual_data.index, forecast_actual_data['MSFT'], label='Actual MSFT', color='orange')
# Add a title and labels
plt.title("Forecasted vs Actual Stock Prices for 2019")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




