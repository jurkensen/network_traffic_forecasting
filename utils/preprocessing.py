import pandas as pd
import numpy as np
import glob
from utils.config import *
from sklearn.preprocessing import MinMaxScaler

def load_data(station_path):
    print("\n===== LOADING DATA =====")
    print(f"\n======= FROM STATION: {station_path.name} =======")
    train_path = station_path / f'{station_path.name}.csv'
    test_path = station_path / f'{station_path.name}_test.csv'
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    df_train['time'] = pd.to_datetime(df_train['time'])
    df_test['time'] = pd.to_datetime(df_test['time'])

    df_train = df_train.set_index('time')
    df_test = df_test.set_index('time')

    df_train = df_train.sort_index()
    df_test = df_test.sort_index()
    # Check for frequency
    infer_freq_train = pd.infer_freq(df_train.index)
    if infer_freq_train:
        # Ensure the index has a frequency
        df_train = df_train.asfreq(infer_freq_train)
        df_test = df_test.asfreq(infer_freq_train)
    
    df_train, df_test = handle_missing_values(df_train, df_test)
    return df_train, df_test


def initial_data_stats(df_train, df_test, station_path):
    # Basic information
    print(f'\nStats for data gathered at station {station_path.name}:')
    print(f'Training data size: {len(df_train)} records')
    print(f'Test data size: {len(df_test)} records') 
    # Train/test split ratio
    total_size = len(df_train) + len(df_test)
    train_ratio = len(df_train) / total_size * 100
    test_ratio = len(df_test) / total_size * 100
    print(f'Train/Test split: {train_ratio:.1f}% / {test_ratio:.1f}%')
    
    # Column information
    print(f'\nColumns ({len(df_train.columns)}): {list(df_train.columns)}')
    

    print('For each site, eleven features are provided related to downlink and uplink traffic. ' \
    'Specifically, at each timestep the measurements concern the ' \
    'DownLink and UpLink traffic,' \
    ' the number of RNTIs (RNTI Count),' \
    ' the number of allocated resource blocks (RB Up and RB Down)'
    ' and their variances (RB Up Var and RB Down Var), ' \
    'the Modulation and Coding Schemes (MCS) (MCS Up and MCS Down)'
    ' and their variances (MCS Up Var and MCS Down Var). ' \
    'Each timestep represents an aggregation of these measurements into a two-minute interval.')
    
    # Data types
    print('\nData types:')
    print(df_train.dtypes)
    
    # Check for missing values
    train_missing = df_train.isnull().sum().sum()
    test_missing = df_test.isnull().sum().sum()
    print(f'\nMissing values: {train_missing} in training, {test_missing} in test')
    
    # Time range information (assuming there's a timestamp column)
    time_cols = [col for col in df_train.columns if 'time' in col.lower()]
    if time_cols:
        time_col = time_cols[0]
        try:
            df_train[time_col] = pd.to_datetime(df_train[time_col])
            df_test[time_col] = pd.to_datetime(df_test[time_col])
            print(f'\nTime range:')
            print(f'  Training: {df_train[time_col].min()} to {df_train[time_col].max()}')
            print(f'  Test: {df_test[time_col].min()} to {df_test[time_col].max()}')
        except:
            print(f'\nCould not parse {time_col} as datetime')
    
    # Basic statistics for numeric columns
    print('\nTraining data statistics:')
    print(df_train.describe())

    # Check for duplicates
    train_duplicates = df_train.duplicated().sum()
    test_duplicates = df_test.duplicated().sum()
    print(f'\nDuplicate records: {train_duplicates} in training, {test_duplicates} in test')


def handle_missing_values(train_series, test_series):
    clean_train_series = train_series.ffill().bfill()
    clean_test_series = test_series.ffill().bfill()
    return clean_train_series, clean_test_series


def aggregate_data(df, interval_minutes=1):
    # Create the resampling rule string (e.g., '15min', '1H')
    rule = f'{interval_minutes}H'
    
    # Perform resampling with mean aggregation for all columns
    aggregated_df = df.resample(rule).mean()
        
    print(f"Original data points: {len(df)}")
    print(f"Aggregated data points: {len(aggregated_df)}")
    print(f"Reduction ratio: {len(aggregated_df)/len(df):.2f}")
    
    return aggregated_df

def normalize_column(df, column_name, feature_range=(0,1)):
    """Normalize a single column using MinMaxScaler"""
    scaler = MinMaxScaler(feature_range=feature_range)
    # Get the column as a DataFrame to preserve index
    column_data = df[[column_name]]
    # Fit and transform
    normalized_data = scaler.fit_transform(column_data)
    # Return as DataFrame with original index
    normalized_column = pd.DataFrame(normalized_data, index=df.index, columns=[column_name])
    return normalized_column, scaler

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

def preprocess_outliers(df, cap_percentile=99, floor_percentile=1):
    """
    Preprocess data using flooring and capping technique 
    as described in the paper
    """
    df_processed = df.copy()
    for col in df_processed.columns:
        # Get floor and cap values (1st and 99th percentiles)
        floor_val = df_processed[col].quantile(floor_percentile/100)
        cap_val = df_processed[col].quantile(cap_percentile/100)
        
        df_processed[col] = df_processed[col].clip(lower=floor_val, upper=cap_val)
    
    return df_processed