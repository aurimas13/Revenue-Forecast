<p align=center>
  <img height="350px" src="https://github.com/aurimas13/Revenue-Forecast/blob/main/public/images/revenue.jpg"/>
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

# **Overview** </b>

The Revenue Forecast project aimed to predict daily and monthly revenues using time series data and weather features by comparing the performance of three different models: Long Short-Term Memory (LSTM), Prophet, and Autoregressive Integrated Moving Average (ARIMA). The main tasks performed in the notebooks were as follows:

1. **Data Preparation**: The time series data and weather features were loaded, cleaned, and preprocessed. Missing values were imputed, and the data was divided into training and testing sets.

2. **LSTM Model**: An LSTM model was developed using PyTorch to predict revenue based on the time series data and weather features. The model was trained and evaluated using the test data.

3. **Prophet Model**: The Prophet model from Facebook was utilized, which is designed for multivariate time series forecasting with the ability to incorporate additional regressors. The model was trained on the revenue time series data and weather features, and its performance was evaluated using the test data.

4. **ARIMA Model**: An ARIMA model was implemented for univariate time series forecasting. The model was trained on the revenue time series data, and its performance was evaluated using the test data.

5. **Comparing Models**: The performance of all three models was compared visually and quantitatively using metrics such as MSE, MAE, RMSE, and MAPE. The comparison allowed for the identification of the model with the best predictive accuracy.

6. **Tuning Models**: The ARIMA and Prophet models were further tuned to improve their performance. The auto_arima function and manual inspection was used to identify the best parameters for the ARIMA model, while the Prophet model was refined by incorporating additional regressors.

7. **Comparing Tuned Models**: The performance of the tuned ARIMA and Prophet models was compared using the same metrics as before. The comparison enabled the identification of the best model for predicting revenue in this specific case.

Based on the comparison of accuracy metrics, the ARIMA model was found to be the best choice for this particular dataset and problem, despite its limitations in handling multivariate data. This analysis demonstrated the importance of exploring multiple models and metrics when making predictions and highlights the need for model selection to be tailored to the specific dataset and problem at hand.

# Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

# Requirements

- Python 3.7+
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
2. Create a virtual environment:
```
python3 -m venv venv
```
3. Activate the virtual environment:
```
source venv/bin/activate
```
4. Install the required packages:
```
pip install -r requirements.txt
```

# Usage

1. Add your historical revenue and weather or just revenue data to a CSV file with columns Date and Revenue for Daily or Monthly Forecast.
2. Update the file path in the code to point to your CSV file.
3. Run the main script:
```
python main.py
```
4. The script will preprocess the data, train the ARIMA, LSTM, and Prophet models, and evaluate their performance.
5. The results will be displayed, including visualizations of the model predictions and a comparison of their performance metrics.

# Results

The repository includes a comparison of the ARIMA, LSTM, and Prophet models' performance in terms of MSE, MAE, MAPE, and RMSE. The comparison will help you decide which model is best suited for your revenue forecasting needs.

# Contributing

Contributions are welcome! Please submit a pull request or create an issue to discuss any changes or improvements you would like to make.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/aurimas13/Revenue-Forecast/blob/main/LICENSE) file for details.




