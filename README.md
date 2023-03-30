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

### Notebooks

The Revenue Forecast project aimed to predict daily and monthly revenues using time series data and weather features by comparing the performance of three different models: Long Short-Term Memory (LSTM), Prophet, and Autoregressive Integrated Moving Average (ARIMA). The main tasks performed in the notebooks were as follows:

1. **Data Preparation**: The time series data and weather features were loaded, cleaned, and preprocessed. Missing values were imputed, and the data was divided into training and testing sets.

2. **LSTM Model**: An LSTM model was developed using PyTorch to predict revenue based on the time series data and weather features. The model was trained and evaluated using the test data.

3. **Prophet Model**: The Prophet model from Facebook was utilized, which is designed for multivariate time series forecasting with the ability to incorporate additional regressors. The model was trained on the revenue time series data and weather features, and its performance was evaluated using the test data.

4. **ARIMA Model**: An ARIMA model was implemented for univariate time series forecasting. The model was trained on the revenue time series data, and its performance was evaluated using the test data.

5. **Comparing Models**: The performance of all three models was compared visually and quantitatively using metrics such as MSE, MAE, RMSE, and MAPE. The comparison allowed for the identification of the model with the best predictive accuracy.

6. **Tuning Models**: The ARIMA and Prophet models or just ARIMA model were further tuned to improve their performance. The auto_arima function and manual inspection was used to identify the best parameters for the ARIMA model, while the Prophet model was refined by incorporating additional regressors.

7. **Comparing Tuned Models**: The performance of the tuned ARIMA and Prophet models or just ARIMA model was compared using the same metrics as before. The comparison enabled the identification of the best model for predicting revenue in this specific case.

Based on the comparison of accuracy metrics, the ARIMA model was found to be the best choice for this particular dataset and problem, despite its limitations in handling multivariate data. This analysis demonstrated the importance of exploring multiple models and metrics when making predictions and highlights the need for model selection to be tailored to the specific dataset and problem at hand.

### API

The Revenue Forecast project also provides an API to forecast monthly revenue for a business using historical revenue and weather data. The API uses a time series forecasting method, specifically the ARIMA model from analysis of notebooks, to predict the revenue for the next month. The data is preprocessed, and features like wind and weather conditions are converted into numerical values. The ARIMA model is trained on the preprocessed data, and the prediction is provided as a JSON object when requested.

The Flask app serves as the API for the project, allowing users to either GET or POST requests to the `/forecast` endpoint. When a GET request is made, the API returns the predicted revenue for the next month using the cached data. A POST request allows users to upload new data, which is then used to update the cached data and make a prediction based on the updated dataset.

The 2nd part of the project is designed to be easy to use and integrate into various applications where monthly revenue forecasts are required. The API can be extended to include additional features and models to improve the accuracy and usefulness of the predictions.

To get started, follow the Installation and Usage sections in the documentation.

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
- [To Do](#to-do)
- [License](#license)

# Getting Started

Follow these instructions to set up the project on your local machine for development and testing purposes.

## Requirements

To run the porject Python 3.7+ is needswhile libraries required are at [requirments](https://github.com/aurimas13/Revenue-Forecast/blob/main/requirements.txt) file.

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

6. Run the monthly_revenue_forecast.py script to create and save the ARIMA model:

```
python monthly_revenue_forecast.py
```

7. Start the Flask development server:

```
flask run
```

# Usage

***GET request***

To get the predicted revenue for the next month using the cached data, send a GET request to the `/forecast` endpoint<sup>1</sup>:

```
curl -X GET http://localhost:5000/forecast
```

The API will return a JSON object containing the predicted revenue:

```
{
  "prediction": 231099270
}
```

***POST request***

To get the predicted revenue for the next month after uploading new excel data, send a POST request to the /`forecast` endpoint with the file as form-data<sup>2</sup>:

```
curl -X POST -H "Content-Type: multipart/form-data" -F "file=@/absolute/path/to/November_data.xlsx" http://localhost:5000/forecast
```

The API will return a JSON object containing the predicted revenue:

```
{
  "prediction": 249284462
}
```

<sup>1</sup> GET  - slower
<sup>2</sup> POST - faster

# Project Structure

The project is organized as follows:

- `app.py`: The main Flask API file that initializes the ARIMA model and sets up the /forecast endpoint.
- `Monthly_Revenue_Forecast.py`: This module contains functions for preprocessing the data, making predictions using the ARIMA model, and creating and saving the ARIMA model.
- `arima_results.pkl`: The saved ARIMA model.
- `requirements.txt`: The list of required packages for the project.

# Docker

## Prerequisites

- Docker installed on your system. Follow the [official Docker documentation](https://docs.docker.com/engine/install/) for installation instructions.

**Steps**

1. Open a terminal and navigate to the project directory where Dockerfile is after cloning repo from [Installation](#installation).
2. Build the Docker image by running the following command in the terminal:

```
docker build -t revenue-forecast .
```

3. Once the image is built, run a container using the following command:

```
docker run -p 5000:5000 --name revenue-forecast-container revenue-forecast
```

4. The Flask app should now be running on your local machine at <http://localhost:5000>. You can access the /forecast endpoint by sending a GET or POST request using the instructions provided in the project documentation.

To run the app then go and follow what is said at [Usage](#usage).

# Results

The repository includes a comparison of the ARIMA, LSTM, and Prophet models' performance in terms of MSE, MAE, MAPE, and RMSE. The comparison in the Jupyter notebooks will help you decide which model is best suited for your revenue forecasting needs. Default chosen is ARIMA model for both Daily and Monthly reevenue forecasts.

The prediction results may vary depending on the data used for training and forecasting. The example values provided for GET and POST requests are based on the provided data, and you may see different results when using your own data.

Make sure to replace the example values in the documentation with the actual values when using your data.

# Contributing

Contributions are welcome! Please submit a pull request or create an issue to discuss any changes or improvements you would like to make.

# To Do

- Make the GET response faster by creating another endpoint.
- Implement Frontend.
- Creat several more endponts like `/` that explains what needs to be done.
- Optimize Prophet model for multivariate analysis.
- Create API for daily revenue forecast.

# License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/aurimas13/Revenue-Forecast/blob/main/LICENSE) file for details.
