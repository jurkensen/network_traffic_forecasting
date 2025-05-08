import pandas as pd
import numpy as np
import glob
from utils.config import *

def load_data(station_path):
    train_path = station_path / f'{station_path.name}.csv'
    test_path = station_path / f'{station_path.name}_test.csv'
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
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
    time_cols = [col for col in df_train.columns if 'time' in col.lower() or 'date' in col.lower()]
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


def handle_missing_values(df_train, df_test):
    print(len(df_train), len(df_test))
    missing_cols_train = df_train.columns[df_train.isnull().any()].tolist()
    missing_cols_test = df_test.columns[df_test.isnull().any()].tolist()
    
    missing_counts_train = df_train[missing_cols_train].isnull().sum()
    missing_counts_test = df_test[missing_cols_test].isnull().sum()
    print(missing_cols_train, missing_counts_train)
    print(missing_cols_test, missing_counts_test)