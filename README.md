<p align=center>
  <img height="400px" src="https://github.com/aurimas13/Revenue-Forecast/blob/main/public/images/revenue.jpg"/>
</p>

<p align="center" > <b> Revenue Forecast </b> </p>
<p align=center>
<a href="https://img.shields.io/github/last-commit/aurimas13/Revenue-Forecast"><img alt="lastcommit" src="https://img.shields.io/github/last-commit/aurimas13/Revenue-Forecast?style=social"/></a
<a href="https://img.shields.io/github/issues/aurimas13/Revenue-Forecast"><img alt="issues" src="https://img.shields.io/github/issues/aurimas13/Revenue-Forecast?style=social"/></a
<a href="https://img.shields.io/github/stars/aurimas13/Revenue-Forecast"><img alt="stars" src="https://img.shields.io/github/stars/aurimas13/Revenue-Forecast?style=social"/></a
<a href="https://img.shields.io/github/forks/aurimas13/Revenue-Forecast"><img alt="twitter" src="https://img.shields.io/github/forks/aurimas13/Revenue-Forecast?style=social"/></a
  <a href="https://twitter.com/anausedas"><img alt="twitter" src="https://img.shields.io/twitter/follow/anausedas?style=social"/></a>
</p>

# Overview

The Revenue Forecast project aimed to predict daily and monthly revenues using time series data and weather features by comparing the performance of three different models: Long Short-Term Memory (LSTM), Prophet, and Autoregressive Integrated Moving Average (ARIMA). The main tasks performed in the notebooks were as follows:

1. **Data Preparation**: The time series data and weather features were loaded, cleaned, and preprocessed. Missing values were imputed, and the data was divided into training and testing sets.

2. **LSTM Model**: An LSTM model was developed using PyTorch to predict revenue based on the time series data and weather features. The model was trained and evaluated using the test data.

3. **Prophet Model**: The Prophet model from Facebook was utilized, which is designed for multivariate time series forecasting with the ability to incorporate additional regressors. The model was trained on the revenue time series data and weather features, and its performance was evaluated using the test data.

4. **ARIMA Model**: An ARIMA model was implemented for univariate time series forecasting. The model was trained on the revenue time series data, and its performance was evaluated using the test data.

5. **Comparing Models**: The performance of all three models was compared visually and quantitatively using metrics such as MSE, MAE, RMSE, and MAPE. The comparison allowed for the identification of the model with the best predictive accuracy.

6. **Tuning Models**: The ARIMA and Prophet models or just ARIMA model were further tuned to improve their performance. The auto_arima function and manual inspection was used to identify the best parameters for the ARIMA model, while the Prophet model was refined by incorporating additional regressors.

7. **Comparing Tuned Models**: The performance of the tuned ARIMA and Prophet models or just ARIMA model was compared using the same metrics as before. The comparison enabled the identification of the best model for predicting revenue in this specific case.

Based on the comparison of accuracy metrics, the ARIMA model was found to be the best choice for this particular dataset and problem, despite its limitations in handling multivariate data. This analysis demonstrated the importance of exploring multiple models and metrics when making predictions and highlights the need for model selection to be tailored to the specific dataset and problem at hand.

# Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Docker](#docker)
- [Contributing](#contributing)
- [License](#license)

# Getting Started

Follow these instructions to set up the project on your local machine for development and testing purposes.
## Requirements

- Python 3.7+
- Git
- pip
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels
- torch
- fbprophet
- pmdarima
- Jupyter Notebook
 

# Installation

To set up a virtual environment and install the required packages, follow these steps:

1. Clone the repository:
```
git clone https://github.com/aurimas13/Revenue-Forecast.git
```
2. Change the working directory:
```
cd Revenue-Forecast
```
3. Create a virtual environment:
```
python -m venv venv
```
4. Activate the virtual environment:
  - On Windows:
```
venv\Scripts\activate
```
  - On macOS and Linux:
```
source venv/bin/activate
```
5. Install the required packages:
```
pip install -r requirements.txt
```
6. Run the Monthly_Revenue_Forecast.py script to create and save the ARIMA model:
```
python Monthly_Revenue_Forecast.py
```
7. Start the Flask development server:
```
flask run
```
# Usage

***GET request***

To get the predicted revenue for the next month using the cached data, send a GET request to the `/forecast` endpoint:
```curl -X GET http://localhost:5000/forecast```
The API will return a JSON object containing the predicted revenue:
```
{
  "prediction": 231099270.4612668
}
```

***POST request***

To get the predicted revenue for the next month after uploading new excel data, send a POST request to the /`forecast` endpoint with the file as form-data:
```
curl -X POST -H "Content-Type: multipart/form-data" -F "file=@November_data.xlsx" http://localhost:5000/forecast
```
The API will return a JSON object containing the predicted revenue:
```
{
  "prediction": 249284462.98393857
}
```

# Project Structure

The project is organized as follows:

- `app.py`: The main Flask API file that initializes the ARIMA model and sets up the /forecast endpoint.
- `Monthly_Revenue_Forecast.py`: This module contains functions for preprocessing the data, making predictions using the ARIMA model, and creating and saving the ARIMA model.
- `arima_results.pkl`: The saved ARIMA model.
- `requirements.txt`: The list of required packages for the project.

# Docker

To build & run docker do these commands: 
`docker build -t monthly-revenue-prediction .` & `docker run -p 5000:5000 monthly-revenue-prediction`

To run the app then go and follow what is said at [Installation](#installation).

# Results

The repository includes a comparison of the ARIMA, LSTM, and Prophet models' performance in terms of MSE, MAE, MAPE, and RMSE. The comparison in the Jupyter notebooks will help you decide which model is best suited for your revenue forecasting needs. Default chosen is ARIMA model for both Daily and Monthly reevenue forecasts.

The prediction results may vary depending on the data used for training and forecasting. The example values provided for GET and POST requests are based on the provided data, and you may see different results when using your own data.

Make sure to replace the example values in the documentation with the actual values when using your data.

# Contributing

Contributions are welcome! Please submit a pull request or create an issue to discuss any changes or improvements you would like to make.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/aurimas13/Revenue-Forecast/blob/main/LICENSE) file for details.




