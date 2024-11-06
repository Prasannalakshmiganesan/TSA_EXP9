## Developed By: Prasannalakshmi G
## Reg No: 212222240075
## Date: 25.10.2024

# EX.NO.09        A project on Time series analysis on ratings rate forecasting using ARIMA model in python


### AIM:
The aim of this project is to forecast ratings rate in Goodreads Books dataset using the ARIMA model and evaluate its accuracy through visualization and statistical metrics.

### ALGORITHM:
Here's a condensed 5-point version for applying ARIMA on the Goodreads dataset:

1. Load and Prepare Data: Load `Goodreads_books.csv`, convert `publication_date` to datetime, and set it as the index.

2. Initial Visualization and Stationarity Check: Plot the time series; use ADF, ACF, and PACF to assess stationarity.

3. Apply Differencing: If non-stationary, apply differencing to stabilize the series.

4. Select ARIMA Parameters: Use `auto_arima` or ACF/PACF plots to find suitable `(p, d, q)` values.

5. Fit Model, Forecast, and Evaluate: Fit the ARIMA model, make predictions, and evaluate using MAE, RMSE, and comparison plots.

   
### PROGRAM:

```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
data = pd.read_csv('Goodreads_books.csv')
# Convert 'publication_date' (or the relevant date column) to datetime and set as index
data['publication_date'] = pd.to_datetime(data['publication_date'], errors='coerce')
data.set_index('publication_date', inplace=True)

# Filter data from a specific date onward, if needed
data = data[data.index >= '2018-01-01']

# Convert 'average_rating' column (or another relevant column) to numeric and handle missing values
data['average_rating'] = pd.to_numeric(data['average_rating'], errors='coerce')
data['average_rating'].fillna(method='ffill', inplace=True)

# Plot the average rating to inspect for trends
plt.figure(figsize=(10, 5))
plt.plot(data['average_rating'], label='Goodreads Average Rating')
plt.title('Time Series of Goodreads Average Rating')
plt.xlabel('Date')
plt.ylabel('Average Rating')
plt.legend()
plt.show()

# Check stationarity with ADF test
result = adfuller(data['average_rating'].dropna())
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# If p-value > 0.05, apply differencing
data['average_rating_diff'] = data['average_rating'].diff().dropna()
result_diff = adfuller(data['average_rating_diff'].dropna())
print('Differenced ADF Statistic:', result_diff[0])
print('Differenced p-value:', result_diff[1])

# Plot ACF and PACF for differenced data
plot_acf(data['average_rating_diff'].dropna())
plt.title('ACF of Differenced Average Rating')
plt.show()
plot_pacf(data['average_rating_diff'].dropna())
plt.title('PACF of Differenced Average Rating')
plt.show()

# Plot Differenced Representation
plt.figure(figsize=(10, 5))
plt.plot(data['average_rating_diff'], label='Differenced Average Rating', color='red')
plt.title('Differenced Representation of Goodreads Average Rating')
plt.xlabel('Date')
plt.ylabel('Differenced Average Rating')
plt.axhline(0, color='black', lw=1, linestyle='--')
plt.legend()
plt.show()

# Use auto_arima to find the optimal (p, d, q) parameters
stepwise_model = auto_arima(data['average_rating'].dropna(), start_p=1, start_q=1,
                            max_p=3, max_q=3, seasonal=False, trace=True)
p, d, q = stepwise_model.order
print(stepwise_model.summary())

# Fit the ARIMA model using the optimal parameters
model = sm.tsa.ARIMA(data['average_rating'].dropna(), order=(p, d, q))
fitted_model = model.fit()
print(fitted_model.summary())

# Forecast the next 30 periods (adjust frequency based on your data, e.g., monthly, daily, etc.)
forecast = fitted_model.forecast(steps=30)
forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30, freq='M')

# Plot actual vs forecasted values
plt.figure(figsize=(12, 6))
plt.plot(data['average_rating'], label='Actual')
plt.plot(forecast_index, forecast, label='Forecast', color='orange')
plt.xlabel('Date')
plt.ylabel('Average Rating')
plt.title('ARIMA Forecast of Goodreads Average Rating')
plt.legend()
plt.show()

# Evaluate the model with MAE and RMSE
predictions = fitted_model.predict(start=0, end=len(data['average_rating']) - 1)
mae = mean_absolute_error(data['average_rating'], predictions)
rmse = np.sqrt(mean_squared_error(data['average_rating'], predictions))
print('Mean Absolute Error (MAE):', mae)
print('Root Mean Squared Error (RMSE):', rmse)


```

### OUTPUT:
![{D9A4E6FA-A085-4E4D-875B-CB57D30194C5}](https://github.com/user-attachments/assets/bb8731bc-c657-43cd-b06c-26867e5897a8)

![{CC1697FA-3402-43C5-9107-27A7AC33FE97}](https://github.com/user-attachments/assets/3cd8741f-7284-47ae-9014-77b8b3f02282)

![{6431A770-9FFE-419F-AD53-46E55C4CE2BC}](https://github.com/user-attachments/assets/25eba269-e2cf-4d9e-9754-3bd4073199b4)

![{C32ED61F-DD6E-42BE-A8A2-C7D46E9C18A1}](https://github.com/user-attachments/assets/8b802fdc-5ac6-487a-a08c-8396bee42ceb)

![{43D3A39C-F2F5-4940-8466-3ED102A51F23}](https://github.com/user-attachments/assets/1b85fdf0-c425-4592-af9b-d1ae51496c4c)

![{30FB08D8-4FA5-4FC7-A489-DCEE6F0B963B}](https://github.com/user-attachments/assets/a3930bca-912b-4cef-b737-438500e58983)

![{7243DF63-25B0-4006-A0B1-D4DE88C274FC}](https://github.com/user-attachments/assets/180a6b07-2693-4200-b83e-bacec99f19a8)

![{2A2E92E0-014E-4681-A972-236585069FE4}](https://github.com/user-attachments/assets/3b8c8fde-a396-48cf-a30a-a16bd430842f)



### RESULT:
Thus the Time series analysis on Ratings rate in Goodreads Books prediction using the ARIMA model completed successfully.
