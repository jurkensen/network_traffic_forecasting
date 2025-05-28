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
import os
import multiprocessing as mp
from functools import partial

def main():  
    evaluation = interval_study_multiprocessing()
    print("\nFinal evaluation results:")
    print(evaluation)
def process_single_interval(interval_minutes, df_all, target_column='down'):
    """
    Process a single interval - this function will run in parallel
    """
    try:
        print(f"\n===== Training SARIMA model for {interval_minutes} minute intervals =====")
        
        # Create new instance for each process to avoid sharing issues
        sarima = SARIMAPredictor()
        
        # Process data for this interval
        df_agg = aggregate_data(df_all.copy(), interval_minutes)
        df_agg = add_hour_features(df_agg, interval_minutes)    
       
        rows_per_day = int(24 * 60 / interval_minutes)
        rows_per_week = rows_per_day * 7
        
        if len(df_agg) < rows_per_week + rows_per_day:
            print(f"Not enough data for interval {interval_minutes} minutes.")
            return interval_minutes, None
        
        # Select first week for training, next 24h for testing
        train = df_agg.iloc[:rows_per_week]
        test = df_agg.iloc[rows_per_week:rows_per_week + rows_per_day]
        
        # Predict
        raw_predictions, order, seasonal_order = sarima.predict_network_traffic(
            train, test, target_column=target_column
        )
        
        prediction = raw_predictions
        actual = test[target_column]
        
        # Calculate metrics
        mae = mean_absolute_error(actual, prediction)
        mse = mean_squared_error(actual, prediction)
        rmse = np.sqrt(mse)
        
        print(f"ARIMA Order: {order} for interval {interval_minutes} min")
        print(f"Interval {interval_minutes} min - RMSE: {rmse:.4f}")
        print(f"Interval {interval_minutes} min - MSE: {mse:.4f}")
        print(f"Interval {interval_minutes} min - MAE: {mae:.4f}")
        
        # Return results
        result = {
            'rmse': rmse,
            'mae': mae, 
            'mse': mse,
            'order': order,
            'seasonal_order': seasonal_order,
            'actual': actual,
            'prediction': prediction
        }
        
        return interval_minutes, result
        
    except Exception as e:
        print(f"Error processing interval {interval_minutes}: {str(e)}")
        return interval_minutes, None

def interval_study_multiprocessing():
    """
    Main function using multiprocessing
    """
    # Load and preprocess data once
    df_all = load_all_data(POBLESEC_ST)
    df_all = handle_missing_values(df_all)
    df_all = handle_outliers(df_all, 5, 95)
    
    # Define intervals to test
    intervals = [2, 5, 10, 15, 30, 60]
    
    # Get number of CPUs to use
    n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', mp.cpu_count()))
    print(f"Using {n_cpus} CPUs for parallel processing")
    
    # Create partial function with fixed arguments
    process_func = partial(process_single_interval, df_all=df_all, target_column='down')
    
    # Run in parallel
    evaluation = {}
    with mp.Pool(processes=n_cpus) as pool:
        # Process all intervals in parallel
        results = pool.map(process_func, intervals)
        
        # Collect results
        for interval_minutes, result in results:
            if result is not None:
                evaluation[interval_minutes] = result
    
    # Generate plots (do this sequentially to avoid issues)
    print("\nGenerating plots...")
    for interval_minutes in intervals:
        if interval_minutes in evaluation:
            result = evaluation[interval_minutes]
            plot_forecast(
                result['actual'], 
                result['prediction'], 
                result['order'], 
                result['seasonal_order'], 
                target_column='down', 
                interval_minutes=interval_minutes
            )
    
    print("\nFinal evaluation results:")
    print(evaluation)
    return evaluation

# INTERVAL STUDY ON ONE WEEK DATA - TEST 1 DAY AHEAD PREDICTION
def interval_study():
    sarima = SARIMAPredictor()
    df_all = load_all_data(POBLESEC_ST)
    df_all = handle_missing_values(df_all)
    df_all = handle_outliers(df_all, 5, 95) 
    # df_all, scaler = normalize_df_minmax(df_all, feature_range=(0,1))
    
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
        plot_forecast(actual, prediction, order, seasonal_order, target_column='down', interval_minutes=i, file_name=f'sarima_interval_{i}')

    print(evaluation)


if __name__ == '__main__':
    main()