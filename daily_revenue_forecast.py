# Importing several libraries and packages to perform time series forecasting and data analysis. 
import warnings
import os
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import random
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Disable warnings and set logger level
warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

# Set the default figure size
plt.rcParams['figure.figsize'] = (20, 12)

# # <b> Loading data </b>
# 
# Loading two separate datasets of Revenue and Weather from an Excel file hosted on GitHub.

# In[3]:


url = "https://github.com/aurimas13/Revenue-Forecast/blob/main/Dataset/Barbora%20Homework.xlsx?raw=true"
revenue_data = pd.read_excel(url, sheet_name='Revenue', parse_dates=['Date'], engine='openpyxl')
weather_data = pd.read_excel(url, sheet_name='Weather', parse_dates=['dt'], engine='openpyxl')


# Get the top 5 maximum Revenue values
top_5_revenue_values = revenue_data['Revenue'].nlargest(5)

def allowed_file(filename):
    """
    Function to check if the uploaded file has a valid format
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def read_remote_data(url):
    """
    Read revenue and weather data from a remote excel file
    """
    revenue_data = pd.read_excel(url, sheet_name='Revenue', parse_dates=['Date'], engine='openpyxl')
    weather_data = pd.read_excel(url, sheet_name='Weather', parse_dates=['dt'], engine='openpyxl')
    return revenue_data, weather_data

# Removing outliers by defining a function
def is_outlier_iqr(data, column, multiplier=1.5):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return (data[column] < lower_bound) | (data[column] > upper_bound)

def preprocess_data(revenue_data, weather_data):
    """
    Function to preprocess revenue and weather data
    """
        
    # Converting categorical data to numerical data in weather_data for heatmap
    weather_data['wind'] = weather_data['wind'].astype('category').cat.codes
    weather_data['condition'] = weather_data['condition'].astype('category').cat.codes
    weather_data['time'] = weather_data['condition'].astype('category').cat.codes

    # Numeric features in weather_data
    numeric_features = ['temperature', 'dew_point', 'humidity', 'wind_speed', 'pressure', 'precipitation', 'condition', 'wind', 'time']

    # Create a copy of the DataFrame to store the weather cleaned data
    weather_data_cleaned = weather_data.copy()

    # Remove outliers for each numeric feature
    for feature in numeric_features:
        outlier_mask = is_outlier_iqr(weather_data_cleaned, feature)
        weather_data_cleaned = weather_data_cleaned[~outlier_mask]

    # Removing duplicates
    weather_data = weather_data_cleaned.drop_duplicates()
    print(weather_data.duplicated().sum())

    # Remove values of wind, time, condition and precipitation
    weather_data.drop(['wind','time','condition','precipitation'], axis=1, inplace=True)

    # Filter rows that have Revenue less than or equal to 10^7 after inspecting and doing EDA. Simply removing outliers
    revenue_data_cleaned = revenue_data[revenue_data['Revenue'] <= 10**7]
    print(revenue_data_cleaned.isna().sum().sum())


    # Aggregate weather_data on 'dt' to daily level
    daily_weather_data = weather_data.resample('D', on='dt').agg({
        'temperature': 'mean',
        'dew_point': 'mean',
        'humidity': 'mean',
        'wind_speed': 'median',
        'pressure': 'mean',
    }).reset_index()

    # Convert 'dt' column to integer
    daily_weather_data['dt'] = daily_weather_data['dt'].astype(int) // 10**9

    # Fill missing values with linear interpolation
    daily_weather_data.interpolate(method='linear', inplace=True)

    # Convert 'dt' column back to datetime format
    daily_weather_data['dt'] = pd.to_datetime(daily_weather_data['dt'], unit='s')

    window_size = 7

    # Create rolling window features
    daily_weather_data['temperature_7d_avg'] = daily_weather_data['temperature'].rolling(window=window_size).mean()
    daily_weather_data['dew_point_7d_avg'] = daily_weather_data['dew_point'].rolling(window=window_size).mean()
    daily_weather_data['humidity_7d_avg'] = daily_weather_data['humidity'].rolling(window=window_size).mean()
    daily_weather_data['wind_speed_7d_median'] = daily_weather_data['wind_speed'].rolling(window=window_size).median()
    daily_weather_data['pressure_7d_avg'] = daily_weather_data['pressure'].rolling(window=window_size).mean()

    # Fill the NaNs in the first 6 days with the specified values
    daily_weather_data['temperature_7d_avg'].fillna(0, inplace=True, limit=6)
    daily_weather_data['dew_point_7d_avg'].fillna(0, inplace=True, limit=6)
    daily_weather_data['humidity_7d_avg'].fillna(1, inplace=True, limit=6)
    daily_weather_data['pressure_7d_avg'].fillna(981, inplace=True, limit=6)
    daily_weather_data['wind_speed_7d_median'].fillna(14, inplace=True, limit=6)

    # Drop the original columns
    daily_weather_data.drop(columns=['temperature', 'dew_point', 'humidity', 'wind_speed', 'pressure'], inplace=True)

    # Merge revenue_data and daily_weather_data on the Date and dt columns
    merged_data = pd.merge(revenue_data_cleaned, daily_weather_data, left_on='Date', right_on='dt', how='inner')

    # Drop the 'dt' column since it's a duplicate of 'Date'
    merged_data.drop(columns=['dt'], inplace=True)

    # Add the lagged revenue column to the merged_data
    merged_data['lagged_revenue'] = merged_data['Revenue'].shift(1).fillna(0)

    # Replace zeros with NaNs
    merged_data = merged_data.replace(0, np.nan)

    # Drop rows with NaN values
    merged_data = merged_data.dropna()

    return merged_data

def predict_next_month(model, preprocessed_data):
    """
    Predict the next day's revenue
    """
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

    # Split the data into train, validation, and test sets
    train_data = merged_data[merged_data['Date'] < '2021-09-01']
    val_data = merged_data[(merged_data['Date'] >= '2021-09-01') & (merged_data['Date'] < '2022-01-01')]
    test_data = merged_data[merged_data['Date'] >= '2022-04-01']

    # Separate the features (X) and the target (y)
    X_train = train_data.drop(columns=['Date', 'Revenue'])
    y_train = train_data['Revenue']
    X_val = val_data.drop(columns=['Date', 'Revenue'])
    y_val = val_data['Revenue']
    X_test = test_data.drop(columns=['Date', 'Revenue'])
    y_test = test_data['Revenue']


    # Scale the features
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Convert the scaled features back to DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Initialize a StandardScaler instance for the target variable y
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.to_numpy(dtype='float32').reshape(-1, 1))
    y_val_scaled = y_scaler.transform(y_val.to_numpy(dtype='float32').reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.to_numpy(dtype='float32').reshape(-1, 1))

    # Convert DataFrames to NumPy arrays and then to PyTorch tensors
    X_train_numpy = X_train_scaled.to_numpy(dtype='float32')
    X_val_numpy = X_val_scaled.to_numpy(dtype='float32')
    X_test_numpy = X_test_scaled.to_numpy(dtype='float32')
    y_train_numpy = y_train_scaled
    y_val_numpy = y_val_scaled
    y_test_numpy = y_test_scaled
    X_train_tensor = torch.tensor(X_train_numpy)
    X_val_tensor = torch.tensor(X_val_numpy)
    X_test_tensor = torch.tensor(X_test_numpy)
    y_train_tensor = torch.tensor(y_train_numpy)
    y_val_tensor = torch.tensor(y_val_numpy)
    y_test_tensor = torch.tensor(y_test_numpy)


    # Create TensorDatasets and DataLoaders for train and validation sets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create TensorDataset and DataLoader for the test set
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize Neural Net
model = LSTM(input_size, hidden_size, num_layers, output_size, dropout_rate).to(device)

# Define Loss with Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Define number of folds for cross-validation
num_folds = 5

# Initialize list to store model evaluations
evaluations = []

# Create KFold cross-validator
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Loop through each fold
for fold, (train_index, test_index) in enumerate(kf.split(X_train_tensor)):
    print(f'Fold {fold+1} / {num_folds}')
    
    # Split data into train and test sets for this fold
    X_train_fold = X_train_tensor[train_index]
    y_train_fold = y_train_tensor[train_index]
    X_val_fold = X_train_tensor[test_index].to(device)
    y_val_fold = y_train_tensor[test_index]
    
    # Create train and validation loaders for this fold
    train_fold_dataset = TensorDataset(X_train_fold, y_train_fold)
    train_fold_loader = DataLoader(train_fold_dataset, batch_size=batch_size, shuffle=True)

    val_fold_dataset = TensorDataset(X_val_fold, y_val_fold)
    val_fold_loader = DataLoader(val_fold_dataset, batch_size=batch_size, shuffle=False)

    
    # Initialize Net
    model = LSTM(input_size, hidden_size, num_layers, output_size, dropout_rate).to(device)

    # Define Loss with Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train
    for epoch in range(num_epochs):
        for i, (inputs, targets) in enumerate(train_fold_loader):
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.view(-1, 1)  # Reshape the output tensor
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Fold {fold+1} / {num_folds}, Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Model evaluation on validation set
    model.eval()
    y_pred_vals = []
    y_true_vals = []
    with torch.no_grad():
      for inputs, targets in val_fold_loader:
            inputs = inputs.to(device).float()
            targets = targets.cpu().detach().numpy()
            y_pred_val = model(inputs)
            y_pred_val = y_pred_val.cpu().detach().numpy()
            y_pred_vals.extend(y_pred_val)
            y_true_vals.extend(targets)

    y_pred_unscaled_val = y_scaler.inverse_transform(np.array(y_pred_vals))
    y_val_unscaled = y_scaler.inverse_transform(np.array(y_true_vals))

    mse = mean_squared_error(y_val_unscaled, y_pred_unscaled_val)
    mae = mean_absolute_error(y_val_unscaled, y_pred_unscaled_val)
    mape = mean_absolute_percentage_error(y_val_unscaled, y_pred_unscaled_val)
    rmse = calculate_rmse(y_val_unscaled, y_pred_unscaled_val)

    print(f'Fold {fold+1} / {num_folds}, Validation Set MSE: {mse:.0f}')
    print(f'Fold {fold+1} / {num_folds}, Validation Set MAE: {mae:.0f}')
    print(f'Fold {fold+1} / {num_folds}, Validation Set MAPE: {mape:.2f}')
    print(f'Fold {fold+1} / {num_folds}, Validation Set RMSE: {rmse:.0f}')
    
    evaluations.append({'fold': fold + 1, 'mse': mse, 'mae': mae, 'mape': mape, 'rmse': rmse})


# Train the final model on the entire train set
model.train()
for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device).float()
        targets = targets.to(device).float()

        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.view(-1, 1)  # Reshape the output tensor
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# Model evaluation on the training set
model.eval()
y_pred_train = []
y_true_train = []
with torch.no_grad():
    for inputs, targets in train_loader:
        inputs = inputs.to(device).float()
        targets = targets.cpu().detach().numpy()
        y_pred = model(inputs)
        y_pred = y_pred.cpu().detach().numpy()
        y_pred_train.extend(y_pred)
        y_true_train.extend(targets)

y_pred_unscaled_train = y_scaler.inverse_transform(y_pred_train)
y_train_unscaled = y_scaler.inverse_transform(y_true_train)


# Model evaluation on test set
model.eval()
y_pred_tests = []
y_true_tests = []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device).float()
        targets = targets.cpu().detach().numpy()
        y_pred_test = model(inputs)
        y_pred_test = y_pred_test.cpu().detach().numpy()
        y_pred_tests.extend(y_pred_test)
        y_true_tests.extend(targets)

y_pred_unscaled_test = y_scaler.inverse_transform(y_pred_tests)
y_test_unscaled = y_scaler.inverse_transform(y_true_tests)



# LSTM predictions
lstm_pred = y_pred_unscaled_test.flatten()

# LSTM prediction
input_tensor = torch.tensor(input_values).to(device).unsqueeze(0).float()
with torch.no_grad():
    lstm_prediction_scaled = model(input_tensor).item()

lstm_prediction = y_scaler.inverse_transform([[lstm_prediction_scaled]])[0][0]


# Define Hyperparameters

input_size = X_train_scaled.shape[1]
# hidden_size = 32
hidden_size = 64
# hidden_size = 128
# hidden_size = 256
num_layers = 2
# num_layers = 3
# num_layers = 4
output_size = 1
num_epochs = 100
learning_rate = 0.001
# learning_rate = 0.0001
# batch_size = 16
batch_size = 32
# batch_size = 64
# batch_size = 128
# dropout_rate = 0.2
dropout_rate = 0.3
# dropout_rate = 0.4

# Create TensorDatasets and DataLoaders for train and validation sets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Create TensorDataset and DataLoader for the test set
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# **Initialization**

# In[49]:


# Initialize Neural Net
model = LSTM(input_size, hidden_size, num_layers, output_size, dropout_rate).to(device)

# Define Loss with Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# **Cross-validation**

# In[50]:


# Define number of folds for cross-validation
num_folds = 5

# Initialize list to store model evaluations
evaluations = []

# Create KFold cross-validator
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Loop through each fold
for fold, (train_index, test_index) in enumerate(kf.split(X_train_tensor)):
    print(f'Fold {fold+1} / {num_folds}')
    
    # Split data into train and test sets for this fold
    X_train_fold = X_train_tensor[train_index]
    y_train_fold = y_train_tensor[train_index]
    X_val_fold = X_train_tensor[test_index].to(device)
    y_val_fold = y_train_tensor[test_index]
    
    # Create train and validation loaders for this fold
    train_fold_dataset = TensorDataset(X_train_fold, y_train_fold)
    train_fold_loader = DataLoader(train_fold_dataset, batch_size=batch_size, shuffle=True)

    val_fold_dataset = TensorDataset(X_val_fold, y_val_fold)
    val_fold_loader = DataLoader(val_fold_dataset, batch_size=batch_size, shuffle=False)

    
    # Initialize Net
    model = LSTM(input_size, hidden_size, num_layers, output_size, dropout_rate).to(device)

    # Define Loss with Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train
    for epoch in range(num_epochs):
        for i, (inputs, targets) in enumerate(train_fold_loader):
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.view(-1, 1)  # Reshape the output tensor
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Fold {fold+1} / {num_folds}, Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Model evaluation on validation set
    model.eval()
    y_pred_vals = []
    y_true_vals = []
    with torch.no_grad():
      for inputs, targets in val_fold_loader:
            inputs = inputs.to(device).float()
            targets = targets.cpu().detach().numpy()
            y_pred_val = model(inputs)
            y_pred_val = y_pred_val.cpu().detach().numpy()
            y_pred_vals.extend(y_pred_val)
            y_true_vals.extend(targets)
        # y_pred_val = model(X_val_fold)

    y_pred_unscaled_val = y_scaler.inverse_transform(np.array(y_pred_vals))
    y_val_unscaled = y_scaler.inverse_transform(np.array(y_true_vals))

    mse = mean_squared_error(y_val_unscaled, y_pred_unscaled_val)
    mae = mean_absolute_error(y_val_unscaled, y_pred_unscaled_val)
    mape = mean_absolute_percentage_error(y_val_unscaled, y_pred_unscaled_val)
    rmse = calculate_rmse(y_val_unscaled, y_pred_unscaled_val)

    print(f'Fold {fold+1} / {num_folds}, Validation Set MSE: {mse:.2f}')
    print(f'Fold {fold+1} / {num_folds}, Validation Set MAE: {mae:.2f}')
    print(f'Fold {fold+1} / {num_folds}, Validation Set MAPE: {mape:.2f}')
    print(f'Fold {fold+1} / {num_folds}, Validation Set RMSE: {rmse:.2f}')
    
    evaluations.append({'fold': fold + 1, 'mse': mse, 'mae': mae, 'mape': mape, 'rmse': rmse})


# **Train**

# In[51]:


# Train the final model on the entire train set
model.train() 
for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device).float()
        targets = targets.to(device).float()

        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.view(-1, 1)  # Reshape the output tensor
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# Model evaluation on the training set
model.eval()
y_pred_train = []
y_true_train = []
with torch.no_grad():
    for inputs, targets in train_loader:
        inputs = inputs.to(device).float()
        targets = targets.cpu().detach().numpy()
        y_pred = model(inputs)
        y_pred = y_pred.cpu().detach().numpy()
        y_pred_train.extend(y_pred)
        y_true_train.extend(targets)

y_pred_unscaled_train = y_scaler.inverse_transform(y_pred_train)
y_train_unscaled = y_scaler.inverse_transform(y_true_train)

# Model evaluation on test set
model.eval()
y_pred_tests = []
y_true_tests = []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device).float()
        targets = targets.cpu().detach().numpy()
        y_pred_test = model(inputs)
        y_pred_test = y_pred_test.cpu().detach().numpy()
        y_pred_tests.extend(y_pred_test)
        y_true_tests.extend(targets)

y_pred_unscaled_test = y_scaler.inverse_transform(y_pred_tests)
y_test_unscaled = y_scaler.inverse_transform(y_true_tests)

# Predictions
dates = test_data['Date']
arima_pred = forecast.values
lstm_pred = y_pred_unscaled_test.flatten()


# Model predictions
arima_predictions = arima_pred
lstm_predictions = lstm_pred

# ARIMA prediction
arima_prediction = arima_pred[random_index]
lstm_prediction = lstm_pred[random_index]
