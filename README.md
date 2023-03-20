<p align=center>
  <img height="200px" src="https://github.com/aurimas13/Revenue-Forecast/blob/main/public/images/revenue.jpg"/>
</p>

<p align="center" > <b> Monthly Revenue Forecast </b> </p>
<p align=center>
<a href="https://img.shields.io/github/last-commit/aurimas13/Revenue-Forecast"><img alt="lastcommit" src="https://img.shields.io/github/last-commit/aurimas13/Revenue-Forecast?style=social"/></a
<a href="https://img.shields.io/github/issues/aurimas13/Revenue-Forecast"><img alt="issues" src="https://img.shields.io/github/issues/aurimas13/Revenue-Forecast?style=social"/></a
<a href="https://img.shields.io/github/stars/aurimas13/Revenue-Forecast"><img alt="stars" src="https://img.shields.io/github/stars/aurimas13/Revenue-Forecast?style=social"/></a
<a href="https://img.shields.io/github/forks/aurimas13/Revenue-Forecast"><img alt="twitter" src="https://img.shields.io/github/forks/aurimas13/Revenue-Forecast?style=social"/></a
  <a href="https://twitter.com/anausedas"><img alt="twitter" src="https://img.shields.io/twitter/follow/anausedas?style=social"/></a>
</p>

# Overview

This repository contains the implementation of three time series forecasting models for predicting monthly revenue:

1. Autoregressive Integrated Moving Average (ARIMA)
2. Long Short-Term Memory (LSTM)
3. Facebook Prophet

These models are used to forecast future monthly revenue based on historical data. The repository includes code to preprocess the data, train the models, and evaluate their performance using various metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and Root Mean Squared Error (RMSE).

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
- scikit-learn
- statsmodels
- torch
- fbprophet

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

1. Add your historical revenue data to a CSV file with columns Date and Revenue.
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