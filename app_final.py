import pickle
import pandas as pd
from flask import Flask, request, jsonify, flash
from Monthly_Revenue_Forecast import preprocess_data, predict_next_month, read_remote_data
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['DEBUG'] = True

with open('arima_results.pkl', 'rb') as f:
    model = pickle.load(f)

url = "https://github.com/aurimas13/Revenue-Forecast/blob/main/Dataset/Barbora%20Homework.xlsx?raw=true"

UPLOAD_FOLDER = 'Dataset'
ALLOWED_EXTENSIONS = {'xlsx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Cache the data by reading it once
revenue_data, weather_data = read_remote_data(url)

@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    global revenue_data, weather_data

    if request.method == 'GET':
        
        # Update cached data with the latest data
        revenue_data, weather_data = read_remote_data(url)

        # Preprocess the data
        preprocessed_data = preprocess_data(revenue_data, weather_data)

        # Make a prediction
        prediction = predict_next_month(model, preprocessed_data)

        # Return the prediction as JSON
        return jsonify({'prediction': prediction})
    
    elif request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return jsonify({'error': 'No file uploaded'})

        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return jsonify({'error': 'No selected file'})

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            input_data_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            revenue_input = pd.read_excel(input_data_path, sheet_name='November_Revenue', parse_dates=['Date'], engine='openpyxl')
            weather_input = pd.read_excel(input_data_path, sheet_name='November_Weather', parse_dates=['dt'], engine='openpyxl')

            # Update cached data
            revenue_data = revenue_data.append(revenue_input)
            weather_data = weather_data.append(weather_input)

            # Preprocess the data
            preprocessed_data = preprocess_data(revenue_data.append(revenue_input), weather_data.append(weather_input))

            # Make a prediction
            prediction = predict_next_month(model, preprocessed_data)

            # Return the prediction as JSON
            return jsonify({'prediction': prediction})
        else:
            return jsonify({'error': 'Invalid file format'})

if __name__ == '__main__':
    app.run(debug=True)
