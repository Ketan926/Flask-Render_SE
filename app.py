from flask import Flask, jsonify
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import datetime

app = Flask(__name__)

# Load the pre-trained Bi-LSTM model
bilstm_model1 = tf.keras.models.load_model('ethereum_bilstm_model.keras')
bilstm_model2 = tf.keras.models.load_model('solana_bilstm_model.keras')
# Fetch and preprocess the Ethereum data
def fetch_and_preprocess_data_ethereum():
    eth = yf.Ticker("ETH-USD")
    df = eth.history(period="max")  # Fetch the historical data
    df.drop(columns=['Dividends', 'Stock Splits'], inplace=True)

    # Use 'High', 'Low', 'Open', 'Volume' as features and 'Close' as target
    x = df[['High', 'Low', 'Open', 'Volume']].values
    y = df['Close'].values.reshape(-1, 1)

    # Separate scalers for features and 'Close'
    scaler = MinMaxScaler(feature_range=(0, 1))
    #scaler_y = MinMaxScaler(feature_range=(0, 1))

    # Scale features and target
    x_scaled = scaler.fit_transform(x)
    y_scaled = scaler.fit_transform(y)

    # Create time series data with 60 time steps
    time_steps = 60
    X, Y = create_time_series_data(x_scaled, y_scaled, time_steps)
    
    return df, X, Y, scaler

def fetch_and_preprocess_data_solana():
    sol = yf.Ticker("SOL-USD")
    df = sol.history(period="max")  # Fetch the historical data
    df.drop(columns=['Dividends', 'Stock Splits'], inplace=True)

    # Use 'High', 'Low', 'Open', 'Volume' as features and 'Close' as target
    x = df[['High', 'Low', 'Open', 'Volume']].values
    y = df['Close'].values.reshape(-1, 1)

    # Separate scalers for features and 'Close'
    scaler = MinMaxScaler(feature_range=(0, 1))
    #scaler_y = MinMaxScaler(feature_range=(0, 1))

    # Scale features and target
    x_scaled = scaler.fit_transform(x)
    y_scaled = scaler.fit_transform(y)

    # Create time series data with 60 time steps
    time_steps = 60
    X, Y = create_time_series_data(x_scaled, y_scaled, time_steps)
    
    return df, X, Y, scaler

# Function to create time series data for LSTM
def create_time_series_data(x, y, time_steps=60):
    X, Y = [], []
    for i in range(len(x) - time_steps):
        X.append(x[i:i + time_steps])
        Y.append(y[i + time_steps])
    return np.array(X), np.array(Y)

# Function to predict the next 7 days using the trained Bi-LSTM model
def predict_next_days_ethereum(model, last_sequence, n_days=7):
    predictions = []
    input_sequence_eth = last_sequence.reshape(1,60, X_train_eth.shape[2])  # Reshape to 3D: (1, time_steps, features)

    for _ in range(n_days):
        # Predict the next day's close price
        next_day_pred = model.predict(input_sequence_eth)[0]  # Prediction for the next day (single value)
        # Append the predicted close price
        predictions.append(next_day_pred)

        # Prepare the new input sequence (sliding window):
        # Remove the oldest time step and add the new prediction for the close price
        next_day_pred = np.array([[next_day_pred[0], input_sequence_eth[0, -1, 1], input_sequence_eth[0, -1, 2], input_sequence_eth[0, -1, 3]]])
        input_sequence_eth = np.append(input_sequence_eth[:, 1:, :], next_day_pred[np.newaxis, :, :], axis=1)  # Shift the window

    return np.array(predictions)

def predict_next_days_solana(model, last_sequence, n_days=7):
    predictions = []
    input_sequence_sol = last_sequence.reshape(1,60, X_train_sol.shape[2])  # Reshape to 3D: (1, time_steps, features)

    for _ in range(n_days):
        # Predict the next day's close price
        next_day_pred = model.predict(input_sequence_sol)[0]  # Prediction for the next day (single value)
        # Append the predicted close price
        predictions.append(next_day_pred)

        # Prepare the new input sequence (sliding window):
        # Remove the oldest time step and add the new prediction for the close price
        next_day_pred = np.array([[next_day_pred[0], input_sequence_sol[0, -1, 1], input_sequence_sol[0, -1, 2], input_sequence_sol[0, -1, 3]]])
        input_sequence_sol = np.append(input_sequence_sol[:, 1:, :], next_day_pred[np.newaxis, :, :], axis=1)  # Shift the window

    return np.array(predictions)


# Fetch and preprocess data on API startup
df_eth, X_train_eth, Y_train_eth, scaler_eth = fetch_and_preprocess_data_ethereum()
df_sol, X_train_sol, Y_train_sol, scaler_sol = fetch_and_preprocess_data_solana()

# API endpoint to predict the next 7 days and return last 7 days
@app.route('/', methods=['GET'])
def predict():
    # Get the last sequence of 60 days from the training data for prediction
    last_sequence_eth = X_train_eth[-1]  # Shape: (time_steps, features)
    
    # Predict the next 7 days
    predictions_scaled_eth = predict_next_days_ethereum(bilstm_model1, last_sequence_eth, n_days=7)

    # Inverse transform the predictions to get the original scale
    predictions_original_eth = scaler_eth.inverse_transform(np.concatenate([predictions_scaled_eth, np.zeros((7, 3))], axis=1))[:, 0]

    # Get the actual 'Close' prices for the last 7 days + today
    last_7_days_actual_eth = df_eth['Close'][-8:].values  # Last 7 days + current day

    # Generate future dates for the predicted prices
    last_date_eth = df_eth.index[-1]
    past_dates_eth = [last_date_eth - datetime.timedelta(days=i) for i in range(7,-1,-1)]
    future_dates_eth = [last_date_eth + datetime.timedelta(days=i) for i in range(1, 8)]
    closing_prices_eth= last_7_days_actual_eth.tolist() + predictions_original_eth.flatten().tolist()
    combined_dates_eth = [f"{date.day:02d}-{date.month:02d}-{date.year}" for date in past_dates_eth] + [f"{date.day:02d}-{date.month:02d}-{date.year}" for date in future_dates_eth]
    
    # Get the last sequence of 60 days from the training data for prediction
    last_sequence_sol = X_train_sol[-1]  # Shape: (time_steps, features)
    
    # Predict the next 7 days
    predictions_scaled_sol = predict_next_days_solana(bilstm_model2, last_sequence_sol, n_days=7)

    # Inverse transform the predictions to get the original scale
    predictions_original_sol = scaler_sol.inverse_transform(np.concatenate([predictions_scaled_sol, np.zeros((7, 3))], axis=1))[:, 0]

    # Get the actual 'Close' prices for the last 7 days + today
    last_7_days_actual_sol = df_sol['Close'][-8:].values  # Last 7 days + current day

    # Generate future dates for the predicted prices
    last_date_sol = df_sol.index[-1]
    past_dates_sol = [last_date_sol - datetime.timedelta(days=i) for i in range(7,-1,-1)]
    future_dates_sol = [last_date_sol + datetime.timedelta(days=i) for i in range(1, 8)]
    closing_prices_sol= last_7_days_actual_sol.tolist() + predictions_original_sol.flatten().tolist()
    combined_dates_sol = [f"{date.day:02d}-{date.month:02d}-{date.year}" for date in past_dates_sol] + [f"{date.day:02d}-{date.month:02d}-{date.year}" for date in future_dates_sol]
    
    # Return predictions and actuals in JSON format
    response = {
        'eth_combined_dates':combined_dates_eth,
        'eth_future_dates': [f"{date.day:02d}-{date.month:02d}-{date.year}" for date in future_dates_eth],
        'eth_past_dates': [f"{date.day:02d}-{date.month:02d}-{date.year}" for date in past_dates_eth],
        'eth_last_7_days_actual': last_7_days_actual_eth.tolist(),# Last 7 actual days (including current)
        'eth_predicted_prices': predictions_original_eth.flatten().tolist(),  # Predicted prices for the next 7 days
        'eth_closing_prices': closing_prices_eth,
        'sol_combined_dates':combined_dates_sol,
        'sol_future_dates': [f"{date.day:02d}-{date.month:02d}-{date.year}" for date in future_dates_sol],
        'sol_past_dates': [f"{date.day:02d}-{date.month:02d}-{date.year}" for date in past_dates_sol],
        'sol_last_7_days_actual': last_7_days_actual_sol.tolist(),# Last 7 actual days (including current)
        'sol_predicted_prices': predictions_original_sol.flatten().tolist(),  # Predicted prices for the next 7 days
        'sol_closing_prices': closing_prices_sol
    }
    return jsonify(response)

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)), debug=True)
