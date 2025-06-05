import pandas as pd
import numpy as np
import glob
from utils.config import *
from sklearn.preprocessing import MinMaxScaler

def split_data(df_all, train_size, val_size, test_size):
    train_size = int(len(df_all) * train_size)
    val_size = int(len(df_all) * val_size)
    test_size = int(len(df_all) * test_size)

    df_train = df_all.iloc[:train_size]
    df_val = df_all.iloc[train_size:train_size + val_size]
    df_test = df_all.iloc[-test_size:]

    print(f"Training set size: {len(df_train)}")
    print(f"Validation set size: {len(df_val)}")
    print(f"Test set size: {len(df_test)}")
    
    return df_train, df_val, df_test

def load_all_data(station_path):
    train_path = station_path / f'{station_path.name}.csv'
    test_path = station_path / f'{station_path.name}_test.csv'
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_all = pd.concat([df_train, df_test], axis=0)
    df_all['time'] = pd.to_datetime(df_all['time'])
    df_all = df_all.set_index('time')
    df_all = df_all.sort_index()
    df_all = df_all.asfreq(pd.infer_freq(df_all.index))
    return df_all

def handle_missing_values(df):
    clean_df = df.ffill().bfill()
    return clean_df


def aggregate_data(df, interval_minutes=1):
    # Create the resampling rule string (e.g., '15min', '1H')
    rule = f'{interval_minutes}min'
    
    # Perform resampling with mean aggregation for all columns
    aggregated_df = df.resample(rule).mean()
        
    print(f"Original data points: {len(df)}")
    print(f"Aggregated data points: {len(aggregated_df)}")
    print(f"Reduction ratio: {len(aggregated_df)/len(df):.2f}")
    
    return aggregated_df

def normalize_column(df, column_name, feature_range=(0,1)):
    """Normalize a single column using MinMaxScaler and return the original DataFrame with the normalized column."""
    scaler = MinMaxScaler(feature_range=feature_range)
    column_data = df[[column_name]]
    normalized_data = scaler.fit_transform(column_data)
    # Copy the original DataFrame to avoid modifying in place
    df_normalized = df.copy()
    df_normalized[column_name] = normalized_data
    return df_normalized, scaler

def normalize_df_minmax(df, feature_range=(0, 1)):
    """Normalize a dataframe using MinMaxScaler (default 0-1 range)"""
    scaler = MinMaxScaler(feature_range=feature_range)
    # Store DataFrame index and columns
    index = df.index
    columns = df.columns
    # Fit and transform
    normalized_data = scaler.fit_transform(df)
    # Return as DataFrame with original index and columns
    normalized_df = pd.DataFrame(normalized_data, index=index, columns=columns)
    return normalized_df, scaler

def handle_outliers(df, floor_percentile=10, cap_percentile=90):
    """
    Preprocess data using flooring and capping technique 
    as described in the paper
    """
    df_processed = df.copy()
    for col in df_processed.columns:
        # Get floor and cap values (10th and 90th percentiles)
        if col in ['down', 'up']:
            floor_val = df_processed[col].quantile(floor_percentile/100)
            cap_val = df_processed[col].quantile(cap_percentile/100)
            
            df_processed[col] = df_processed[col].clip(lower=floor_val, upper=cap_val)
    
    return df_processed

def add_hour_features(df, interval_minutes=60):
        """
        Add hour-of-day features to the dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe with datetime index
            
        Returns:
            pd.DataFrame: Dataframe with added hour features
        """                
        # Add hour features
        result = df.copy()
        result['hour'] = result.index.hour
        
        # Add hour as cyclical features (sin/cos transformation)
        # This helps the model understand the circular nature of hours
        hours_in_day = int(1440 / interval_minutes) 
        result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / hours_in_day)
        result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / hours_in_day)
        
        return result
    
def data_pipeline(df_all):
    """
    Main data preprocessing pipeline
    """
    # Handle missing values
    df_all = handle_missing_values(df_all)
    
    df_all = aggregate_data(df_all, interval_minutes=60)
    
    df_all = handle_outliers(df_all)
    
    df_all = add_hour_features(df_all, interval_minutes=1)
        
    return df_all