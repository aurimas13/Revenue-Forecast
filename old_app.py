from flask import Flask, request, jsonify
import Monthly_Revenue_Forecast
import Daily_Revenue_Forecast

app = Flask(__name__)

@app.route('/predict-monthly-revenue', methods=['POST'])
def predict_monthly_revenue():
    data = request.get_json(force=True)
    prediction = Monthly_Revenue_Forecast.predict(data['november_data'])
    response = {
        'prediction': prediction.tolist()
    }
    return jsonify(response)

@app.route('/predict-daily-revenue', methods=['POST'])
def predict_daily_revenue():
    data = request.get_json(force=True)
    prediction = Daily_Revenue_Forecast.predict(data['november_data'])
    response = {
        'prediction': prediction.tolist()
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
