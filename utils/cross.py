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


def process_single_train_split(train_split, df_all, interval_minutes=60, target_column='down'):
    """
    Process a single train split - this function will run in parallel
    """
    try:
        print(f"\n===== Processing train split: {train_split:.2f} =====")
        
        # Create new instance for each process to avoid sharing issues
        sarima = SARIMAPredictor()
        
        # Calculate splits
        test_split = float(1 / 7)  # 1 week - 1 day, 1 day - 1/7 of a day etc.
        n = len(df_all)
        train_size = int(n * train_split)
        test_size = int(train_size * test_split)
        
        if train_size + test_size > n:
            print(f"Not enough data for train_split={train_split}, skipping.")
            return train_split, None
        
        # Create train/test splits
        train = df_all.iloc[:train_size]
        test = df_all.iloc[train_size:train_size + test_size]
        
        # Train and predict
        raw_predictions, order, seasonal_order = sarima.predict_network_traffic(
            train, test, target_column=target_column, interval_minutes=interval_minutes
        )
        
        prediction = raw_predictions
        actual = test[target_column]
        
        # Calculate metrics
        mae = mean_absolute_error(actual, prediction)
        mse = mean_squared_error(actual, prediction)
        rmse = np.sqrt(mse)
        
        print(f"Train split: {train_split:.2f}, Test size: {test_size}, ARIMA Order: {order}")
        print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}")
        
        # Return results
        result = {
            'rmse': rmse,
            'mse': mse,
            'mae': mae,
            'order': order,
            'seasonal_order': seasonal_order,
            'actual': actual,
            'prediction': prediction,
            'test_size': test_size
        }
        
        return train_split, result
        
    except Exception as e:
        print(f"Error processing train_split {train_split}: {str(e)}")
        return train_split, None

def cross_validation_multiprocessing():
    """
    Cross validation using multiprocessing
    """
    interval_minutes = 60  # Use 60 minutes for aggregation
    
    # Load and preprocess data once
    df_all = load_all_data(POBLESEC_ST)
    df_all = handle_missing_values(df_all)
    df_all = handle_outliers(df_all, 5, 95)
    df_all = aggregate_data(df_all, interval_minutes)
    df_all = add_hour_features(df_all, interval_minutes)    
    
    # Define train splits to test
    train_splits = [0.2, 0.4, 0.6, 0.8]
    
    # Get number of CPUs to use
    n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', mp.cpu_count()))
    print(f"Using {n_cpus} CPUs for parallel processing")
    
    # Create partial function with fixed arguments
    process_func = partial(
        process_single_train_split, 
        df_all=df_all, 
        interval_minutes=interval_minutes, 
        target_column='down'
    )
    
    # Run in parallel
    evaluation = {}
    with mp.Pool(processes=n_cpus) as pool:
        # Process all train splits in parallel
        results = pool.map(process_func, train_splits)
        
        # Collect results
        for train_split, result in results:
            if result is not None:
                evaluation[train_split] = result
    
    # Generate plots (do this sequentially to avoid matplotlib issues)
    print("\nGenerating plots...")
    for train_split in train_splits:
        if train_split in evaluation:
            result = evaluation[train_split]
            plot_forecast(
                result['actual'], 
                result['prediction'], 
                result['order'], 
                result['seasonal_order'], 
                target_column='down', 
                interval_minutes=interval_minutes,
                file_name=f'sarima_cv_{train_split:.2f}'
            )
    
    print("\nFinal evaluation results:")
    for train_split in sorted(evaluation.keys()):
        result = evaluation[train_split]
        print(f"Train split: {train_split:.2f} - RMSE: {result['rmse']:.4f}, "
              f"MAE: {result['mae']:.4f}, MSE: {result['mse']:.4f}")
    
    print("\nComplete evaluation dictionary:")
    print(evaluation)
    return evaluation

def cross_validation():
    interval_minutes = 60  # Use 60 minutes for aggregation
    sarima = SARIMAPredictor()
    df_all = load_all_data(POBLESEC_ST)
    df_all = handle_missing_values(df_all)
    df_all = handle_outliers(df_all, 5, 95)
    df_all = aggregate_data(df_all, interval_minutes) 
    df_all = add_hour_features(df_all, interval_minutes)     
    
    evaluation = {}
    test_split = float(1 / 7) # 1 week - 1 day, 1 day - 1/7 of a day etc.
    for train_split in [0.2, 0.4, 0.6, 0.8]:
        n = len(df_all)
        train_size = int(n * train_split)
        test_size = int(train_size * test_split)
        if train_size + test_size > n:
            print(f"Not enough data for train_split={train_split}, skipping.")
            continue

        train = df_all.iloc[:train_size]
        test = df_all.iloc[train_size:train_size + test_size]

        raw_predictions, order, seasonal_order = sarima.predict_network_traffic(train, test, 
            target_column='down', interval_minutes=60)
        prediction = raw_predictions
        actual = test['down']

        mae = mean_absolute_error(actual, prediction)
        mse = mean_squared_error(actual, prediction)
        rmse = np.sqrt(mse)
        print(f"Train split: {train_split:.2f}, Test size: {test_size}, ARIMA Order: {order}")
        print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}")
        plot_forecast(actual, prediction, order, seasonal_order, target_column='down', interval_minutes=interval_minutes, file_name=f'sarima_cv_{train_split:.2f}')
    evaluation[train_split] = {'rmse': rmse, 'mse': mse, 'mae': mae, 'order': order, 'seasonal_order': seasonal_order}
    print(evaluation)
    
evaluation = cross_validation_multiprocessing()
print(evaluation)