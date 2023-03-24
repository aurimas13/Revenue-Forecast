import pickle
import pandas as pd
from mdarima import ARIMA
from flask import Flask, request, jsonify

app = Flask(__name__)

with open('arima_model.pkl', 'rb') as f:
    model = pickle.load(f)

url = "https://github.com/aurimas13/Revenue-Forecast/blob/main/Dataset/Barbora%20Homework.xlsx?raw=true"

revenue_data = pd.read_excel(url, sheet_name='Revenue', parse_dates=['Date'], engine='openpyxl')
weather_data = pd.read_excel(url, sheet_name='Weather', parse_dates=['dt'], engine='openpyxl')

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

def predict_next_month(model, preprocessed_data):
    # Split the data into train and test sets
    train_data = preprocessed_data[preprocessed_data['Date'] < '2022-01-01']
    test_data = preprocessed_data[preprocessed_data['Date'] >= '2022-01-01']

    # Separate the features (X) and the target (y)
    X_train = train_data.drop(columns=['Date', 'Revenue'])
    y_train = train_data['Revenue']
    X_test = test_data.drop(columns=['Date', 'Revenue'])
    y_test = test_data['Revenue']
    # Prepare the time series data for the ARIMA model
    ts_data = preprocessed_data[['Date', 'Revenue']]
    ts_data.set_index('Date', inplace=True)
    
    # Fit the ARIMA model
    p, d, q = 2, 1, 5
    arima_model = ARIMA(ts_data, order=(p, d, q))
    arima_results = arima_model.fit()

    #  Get the forecast for the next months
    num_months_to_forecast = len(test_data)
    forecast = model.forecast(steps=num_months_to_forecast)

    # ARIMA all predictions
    arima_pred = forecast.values

    # ARIMA prediction
    arima_prediction = arima_pred[:-1]

    return arima_prediction

@app.route('/forecast', methods=['POST'])
def forecast():

    # Preprocess the data
    preprocessed_data = preprocess_data(revenue_data, weather_data)

    # Make a prediction
    prediction = predict_next_month(model, preprocessed_data)

    # Return the prediction as JSON
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run()