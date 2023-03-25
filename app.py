import pickle
import pandas as pd
from flask import Flask, request, jsonify
from Monthly_Revenue_Forecast import preprocess_data, predict_next_month

app = Flask(__name__)

with open('arima_results.pkl', 'rb') as f:
    model = pickle.load(f)

url = "https://github.com/aurimas13/Revenue-Forecast/blob/main/Dataset/Barbora%20Homework.xlsx?raw=true"

revenue_data = pd.read_excel(url, sheet_name='Revenue', parse_dates=['Date'], engine='openpyxl')
weather_data = pd.read_excel(url, sheet_name='Weather', parse_dates=['dt'], engine='openpyxl')

@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    if request.method == 'GET':
        # Preprocess the data
        preprocessed_data = preprocess_data(revenue_data, weather_data)

        # Make a prediction
        prediction = predict_next_month(model, preprocessed_data)

        # Return the prediction as JSON
        return jsonify({'prediction': prediction})
    
    elif request.method == 'POST':
        input_data = request.get_json()

        # Convert input_data to pandas dataframes
        revenue_input = pd.DataFrame(input_data['revenue_data'])
        weather_input = pd.DataFrame(input_data['weather_data'])

        # Convert date columns to datetime format
        revenue_input['Date'] = pd.to_datetime(revenue_input['Date'])
        weather_input['dt'] = pd.to_datetime(weather_input['dt'])

        # Preprocess the data
        preprocessed_data = preprocess_data(revenue_data.append(revenue_input), weather_data.append(weather_input))

        # Make a prediction
        prediction = predict_next_month(model, preprocessed_data)

        # Return the prediction as JSON
        return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
