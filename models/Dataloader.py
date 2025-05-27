import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from utils.config import *
from utils.preprocessing import *

class TimeSeriesDataset(Dataset):
    def __init__(self, dataframe, target_columns, seq_length=10, forecast_horizon=1, 
                 exogenous_data=None):
        """
        Create a time series dataset
        
        Args:
            dataframe: Input dataframe with time series data (preprocessed)
            target_columns: List of target column names to predict
            seq_length: Length of input sequences (num_lags)
            forecast_horizon: Number of steps ahead to predict (usually 1)
            exogenous_data: Optional exogenous features dataframe
        """
        self.data = dataframe.values.astype(np.float32)
        self.target_columns = target_columns
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.num_features = len(dataframe.columns)
        
        # Get target column indices
        self.target_indices = [dataframe.columns.get_loc(col) for col in target_columns]
        
        # Handle exogenous data
        self.exogenous_data = None
        if exogenous_data is not None:
            self.exogenous_data = exogenous_data.values.astype(np.float32)
        
        # Pre-compute sequences for efficiency
        self.sequences = []
        self._create_sequences()
    
    def _create_sequences(self):
        """Pre-create all possible sequences"""
        max_idx = len(self.data) - self.seq_length - self.forecast_horizon + 1
        
        for i in range(max_idx):
            # [seq_length, num_features]
            input_seq = self.data[i:i + self.seq_length]
            
            target_idx = i + self.seq_length
            target = self.data[target_idx, self.target_indices]
                        
            exog = None
            if self.exogenous_data is not None:
                exog = self.exogenous_data[target_idx]
            
            self.sequences.append({
                'input': input_seq,
                'target': target,
                'exogenous': exog
            })

    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # [seq_length, num_features]
        X = torch.tensor(seq['input'])

        exogenous = []
        if seq['exogenous'] is not None:
            exogenous = torch.tensor(seq['exogenous'])
                
        y = torch.tensor(seq['target'])
        
        return X, exogenous, y

def create_dataloader(dataframe, target_columns, batch_size=32, 
                               seq_length=10, exogenous_data=None):    
    dataset = TimeSeriesDataset(
        dataframe=dataframe,
        target_columns=target_columns,
        seq_length=seq_length,
        exogenous_data=exogenous_data
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=False
    )


def main():
    # Load the dataset
    df_all = load_all_data(POBLESEC_ST)
    df_all = handle_missing_values(df_all)
    df_all = handle_outliers(df_all, 5, 95) 
    df_all = aggregate_data(df_all, 60)
    df_all, scaler = normalize_df_minmax(df_all, feature_range=(0, 1))
    df_train, df_val, df_test = split_data(df_all, 0.6, 0.2, 0.2) 

main()