from utils.config import *
from utils.preprocessing import *
from utils.plot import plot, plot_forecast
from models.ARIMA import ARIMAPredictor
from models.SARIMAX import SARIMAPredictor
# from models.RNN import *
import matplotlib.pyplot as plt
import pandas as pd 
import pmdarima as pm
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error 


def main():  
    interval_study()


# INTERVAL STUDY ON ONE WEEK DATA - TEST 1 DAY AHEAD PREDICTION
def interval_study():
    sarima = SARIMAPredictor()
    df_all = load_all_data(POBLESEC_ST)
    df_all = handle_missing_values(df_all)
    df_all = handle_outliers(df_all, 5, 95) 
    df_all, scaler = normalize_df_minmax(df_all, feature_range=(0,1))
    
    evaluation = {}
    # Interval Selection - 1 week data
    for i in [2,5,10,15,30,60]:
        print(f"\n===== Training SARIMA model for {i} minute intervals =====")
        df_agg = aggregate_data(df_all.copy(), i)
        df_agg = add_hour_features(df_agg, i)    
        
        rows_per_day = int(24 * 60 / i)
        rows_per_week = rows_per_day * 7

        if len(df_agg) < rows_per_week + rows_per_day:
            print(f"Not enough data for interval {i} minutes.")
            continue

        # Select first week for training, next 24h for testing
        train = df_agg.iloc[ : rows_per_week]
        test = df_agg.iloc[rows_per_week : rows_per_week + rows_per_day]

        raw_predictions, order, seasonal_order = sarima.predict_network_traffic(train, test, target_column='down')
        prediction = raw_predictions  # scaler.inverse_transform(raw_predictions.values.reshape(-1, 1)).flatten()   
        actual = test['down'] # scaler.inverse_transform(test['down'].values.reshape(-1, 1)).flatten()

        mae = mean_absolute_error(actual, prediction)
        rmse = np.sqrt(mean_squared_error(actual, prediction))
        print(f"ARIMA Order: {order} for interval {i} min")
        print(f"Interval {i} min - RMSE: {rmse:.4f}")
        print(f"Interval {i} min - MAE: {mae:.4f}")
        
        evaluation[i] = {rmse: rmse, 'mae': mae, 'order': order}
        plot_forecast(actual, prediction, order, seasonal_order, target_column='down', interval_minutes=i)

    print(evaluation)

def cross_validation():
    arima = ARIMAPredictor()
    df_all = load_all_data(POBLESEC_ST)
    df_all = handle_missing_values(df_all)
    df_all = handle_outliers(df_all, 5, 95)
    df_all = aggregate_data(df_all, 30)  
    
    
    evaluation = {}
    test_split = 0.05
    for train_split in [0.2, 0.4, 0.6, 0.8]:
        n = len(df_all)
        train_size = int(n * train_split)
        test_size = int(n * test_split)
        if train_size + test_size > n:
            print(f"Not enough data for train_split={train_split}, skipping.")
            continue

        train = df_all.iloc[:train_size]
        test = df_all.iloc[train_size:train_size + test_size]['down']

        raw_predictions, order = arima.predict_network_traffic(train, steps=test_size, target_column='down')
        prediction = raw_predictions
        actual = test

        mae = mean_absolute_error(actual, prediction)
        rmse = np.sqrt(mean_squared_error(actual, prediction))
        print(f"Train split: {train_split:.2f}, Test size: {test_size}, ARIMA Order: {order}")
        print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        plot_forecast(actual, prediction, order, target_column='down', interval_minutes=30)
    evaluation[train_split] = {'rmse': rmse, 'mae': mae, 'order': order}
    print(evaluation)

if __name__ == '__main__':
    main()