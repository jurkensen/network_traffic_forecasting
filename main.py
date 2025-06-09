from utils.config import *
from utils.preprocessing import *
from models.SARIMAX import SARIMAPredictor
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.metrics import mean_absolute_error, mean_squared_error 
import os
import multiprocessing as mp
from functools import partial

def main():  
    evaluation = interval_study_multiprocessing()
    
    # Print organized results
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*80)
    
    intervals = [5, 15, 30, 60]
    training_sizes = ['1_day', '1_week', '2_weeks']
    
    # Print summary table
    print(f"{'Interval':<10}{'Training':<12}{'RMSE':<10}{'MAE':<10}{'MSE':<10}{'Order':<15}")
    print("-" * 80)
    
    for interval in intervals:
        for training_size in training_sizes:
            key = (interval, training_size)
            if key in evaluation and evaluation[key] is not None:
                result = evaluation[key]
                print(f"{interval:<10}{training_size:<12}{result['rmse']:<10.4f}{result['mae']:<10.4f}{result['mse']:<10.4f}{str(result['order']):<15}")
            else:
                print(f"{interval:<10}{training_size:<12}{'FAILED':<10}{'FAILED':<10}{'FAILED':<10}{'FAILED':<15}")
    
    print("\n" + "="*80)

def process_single_experiment(params, df_all, target_column='down'):
    """
    Process a single experiment (interval + training_size combination)
    """
    interval_minutes, training_size = params
    
    try:
        print(f"\n===== Training SARIMA: {interval_minutes}min interval, {training_size} training =====")
        
        # Create new instance for each process
        sarima = SARIMAPredictor()
        
        # Process data for this interval
        df_agg = aggregate_data(df_all.copy(), interval_minutes)
        df_agg = add_hour_features(df_agg, interval_minutes)    
       
        rows_per_day = int(24 * 60 / interval_minutes)
        rows_per_week = rows_per_day * 7
        
        # Define training sizes in rows
        training_rows = {
            '1_day': rows_per_day,
            '1_week': rows_per_week, 
            '2_weeks': rows_per_week * 2
        }
        
        train_size = training_rows[training_size]
        test_size = rows_per_week  # Always 1 week test
        
        # Check if we have enough data
        if len(df_agg) < train_size + test_size:
            print(f"Not enough data for {interval_minutes}min interval with {training_size} training.")
            return (interval_minutes, training_size), None
        
        # Split data: training period + 1 week test
        train = df_agg.iloc[:train_size]
        test = df_agg.iloc[train_size:train_size + test_size]
        
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
        
        print(f"SARIMA Order: {order} x {seasonal_order} for {interval_minutes}min, {training_size}")
        print(f"{interval_minutes}min, {training_size} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        # Return results
        result = {
            'rmse': rmse,
            'mae': mae, 
            'mse': mse,
            'order': order,
            'seasonal_order': seasonal_order,
            'actual': actual,
            'prediction': prediction,
            'training_size': training_size,
            'interval': interval_minutes
        }
        
        return (interval_minutes, training_size), result
        
    except Exception as e:
        print(f"Error processing {interval_minutes}min, {training_size}: {str(e)}")
        return (interval_minutes, training_size), None

def interval_study_multiprocessing():
    """
    Main function using multiprocessing for all combinations
    """
    # Load and preprocess data once
    df_all = load_all_data(POBLESEC_ST)
    df_all = handle_missing_values(df_all)
    df_all = handle_outliers(df_all, 5, 95)
    
    # Define study parameters
    intervals = [5, 15, 30, 60]
    training_sizes = ['1_day', '1_week', '2_weeks']
    
    # Create all combinations
    all_combinations = [(interval, training_size) 
                       for interval in intervals 
                       for training_size in training_sizes]
    
    print(f"Running {len(all_combinations)} experiments in parallel...")
    
    # Get number of CPUs to use
    n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', mp.cpu_count()))
    print(f"Using {n_cpus} CPUs for parallel processing")
    
    # Create partial function with fixed arguments
    process_func = partial(process_single_experiment, df_all=df_all, target_column='down')
    
    # Run in parallel
    evaluation = {}
    with mp.Pool(processes=n_cpus) as pool:
        # Process all combinations in parallel
        results = pool.map(process_func, all_combinations)
        
        # Collect results
        for (interval_minutes, training_size), result in results:
            if result is not None:
                evaluation[(interval_minutes, training_size)] = result
    
    return evaluation

if __name__ == '__main__':
    main()