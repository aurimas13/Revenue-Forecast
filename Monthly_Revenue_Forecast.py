import warnings
import os
import pickle
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ValueWarning

# Suppress runtime and value warnings
os.environ['PYTHONWARNINGS'] = 'ignore::RuntimeWarning'
warnings.filterwarnings("ignore", category=ValueWarning)

# Allowed file extensions for input data
ALLOWED_EXTENSIONS = {'xlsx'}

# Function to check if the uploaded file has a valid format
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to read remote data
def read_remote_data(url):
    # Read revenue and weather data from a remote excel file
    revenue_data = pd.read_excel(url, sheet_name='Revenue', parse_dates=['Date'], engine='openpyxl')
    weather_data = pd.read_excel(url, sheet_name='Weather', parse_dates=['dt'], engine='openpyxl')
    return revenue_data, weather_data

# Function to preprocess revenue and weather data
def preprocess_data(revenue_data, weather_data):
    # Drop the 'time' column from the weather_data
    weather_data = weather_data.drop('time', axis=1)

    # Convert categorical features 'wind' and 'condition' to numerical values
    wind_categories = weather_data['wind'].astype('category').cat.categories
    condition_categories = weather_data['condition'].astype('category').cat.categories
    weather_data['wind'] = weather_data['wind'].astype('category').cat.codes
    weather_data['condition'] = weather_data['condition'].astype('category').cat.codes

    # Reseample revenue_data
    revenue_data['Date'] = pd.to_datetime(revenue_data['Date'])
    revenue_data.set_index('Date', inplace=True)
    monthly_revenue_data = revenue_data.resample('M').sum().reset_index()

    # Aggregate weather_data on 'dt' to monthly level
    monthly_weather_data = weather_data.resample('M', on='dt').agg({
        'temperature': 'mean',
        'dew_point': 'mean',
        'humidity': 'mean',
        'wind_speed': 'mean',
        'pressure': 'mean',
        'precipitation': 'sum',
        'wind': 'median',
        'condition': 'median'
    }).reset_index()

    # Merge revenue_data and monthly_weather_data on the Date and dt columns
    merged_data = pd.merge(monthly_revenue_data, monthly_weather_data, left_on='Date', right_on='dt', how='inner')

    # Drop the 'dt' column since it's a duplicate of 'Date'
    merged_data.drop(columns=['dt'], inplace=True)

    # Add the lagged revenue column to the merged_data
    merged_data['lagged_revenue'] = merged_data['Revenue'].shift(1).fillna(0)

    # Replace zeros with NaNs
    merged_data = merged_data.replace(0, np.nan)

    # Drop column 'precipitation' with NaNs values
    merged_data = merged_data.drop('precipitation', axis=1)

    # Drop rows with NaN values
    merged_data = merged_data.dropna()

    # Filter rows that have Revenue less than or equal to 10^9
    merged_data = merged_data[merged_data['Revenue'] <= 10**9]

    return merged_data

# Predict the next month's revenue
def predict_next_month(model, preprocessed_data):
    # Split the data into train and test sets
    train_data = preprocessed_data[preprocessed_data['Date'] < '2022-01-01']
    test_data = preprocessed_data[preprocessed_data['Date'] >= '2022-01-01']

    # Prepare the time series data for the ARIMA model
    ts_data = train_data[['Date', 'Revenue']]
    ts_data.set_index('Date', inplace=True)
    
    # Get the forecast for the next month
    num_months_to_forecast = len(test_data) + 1
    forecast = model.forecast(steps=num_months_to_forecast)

    # Extract the last predicted value for the next month's revenue
    prediction = forecast.values[-1]

    return prediction

# Create and save the ARIMA model
def create_and_save_arima_model(ts_data, p=2, d=1, q=5):
    # Create and fit the ARIMA model
    arima_model = ARIMA(ts_data, order=(p, d, q))
    arima_results = arima_model.fit()

    # Save the fitted ARIMA model
    with open('arima_results.pkl', 'wb') as f:
        pickle.dump(arima_results, f)


if __name__ == "__main__":
    url = "https://github.com/aurimas13/Revenue-Forecast/blob/main/Dataset/Barbora%20Homework.xlsx?raw=true"

    # Read the remote data using the read_remote_data function
    revenue_data, weather_data = read_remote_data(url)

    # Preprocess the revenue and weather data
    preprocessed_data = preprocess_data(revenue_data, weather_data)

    # Split the preprocessed data into training and testing sets
    train_data = preprocessed_data[preprocessed_data['Date'] < '2022-01-01']
    ts_data = train_data[['Date', 'Revenue']]
    ts_data.set_index('Date', inplace=True)

    # Create and save the ARIMA model
    create_and_save_arima_model(ts_data)

    # Load the ARIMA model
    with open('arima_results.pkl', 'rb') as f:
        model = pickle.load(f)
