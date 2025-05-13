import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from utils.config import *
from utils.preprocessing import *

class NetworkDataset(Dataset):
    def __init__(self, dataframe, target_column):
        self.df = dataframe
        self.target_col = target_column
        self.feature_cols = list(self.df.columns)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # access the data in the given index
        row = self.df.iloc[idx]
        features = row.values.astype(np.float32)
        target = row[self.target_column]
        timestamp = self.df.index[idx]
        
        # TODO: do we need the idx? why?
        return {
            'features': torch.tensor(features),
            'target': torch.tensor(target, dtype=torch.float32),
            'timestamp': timestamp,
            'idx': idx
        }
    
def time_series_collate_fn(batch, seq_length=6, forecast_horizon=1):
    """
    Custom collate function that forms sequences from individual timesteps
    
    Args:
        batch: List of samples from the dataset
        seq_length: Length of input sequences
        forecast_horizon: How many steps ahead to predict
    """
    # Sort batch by idx to ensure time order
    batch.sort(key=lambda x: x['idx'])
    
    # Check if we can form a valid sequence
    if len(batch) < seq_length + forecast_horizon:
        raise ValueError(f"Batch size ({len(batch)}) is smaller than required sequence length + forecast horizon ({seq_length + forecast_horizon})")
    
    # Prepare lists for sequences
    feature_seqs = []
    target_seqs = []
    timestamp_seqs = []
    target_timestamps = []
    
    # Create sliding window sequences
    for i in range(len(batch) - seq_length - forecast_horizon + 1):
        # Input sequence
        seq_items = batch[i:i+seq_length]
        features = torch.stack([item['features'] for item in seq_items])
        timestamps = [item['timestamp'] for item in seq_items]
        
        # Target sequence or value
        if forecast_horizon == 1:
            target = batch[i+seq_length]['target'].unsqueeze(0)  # Single step forecast
            target_timestamp = batch[i+seq_length]['timestamp']
        else:
            target_items = batch[i+seq_length:i+seq_length+forecast_horizon]
            target = torch.stack([item['target'] for item in target_items])
            target_timestamp = [item['timestamp'] for item in target_items]
        
        feature_seqs.append(features)
        target_seqs.append(target)
        timestamp_seqs.append(timestamps)
        target_timestamps.append(target_timestamp)
    
    # Stack into batch tensors
    batch_features = torch.stack(feature_seqs)
    batch_targets = torch.stack(target_seqs)
    
    return {
        'features': batch_features,  # Shape: [batch_size, seq_length, n_features]
        'target': batch_targets,     # Shape: [batch_size, forecast_horizon]
        'timestamps': timestamp_seqs,
        'target_timestamps': target_timestamps
    }

def create_dataloader(dataframe, batch_size=32, seq_length=6, forecast_horizon=1, shuffle=False):
    """Create DataLoader with time series support"""
    
    # Create dataset
    dataset = NetworkDataset(dataframe, target_column='down')
    
    # Define custom collate function with closure to set parameters
    collate_fn = lambda batch: time_series_collate_fn(
        batch, 
        seq_length=seq_length,
        forecast_horizon=forecast_horizon
    )
    
    # Use SequentialSampler to preserve time ordering
    sampler = SequentialSampler(dataset)
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size + seq_length + forecast_horizon - 1,  # Adjusted batch size
        sampler=sampler if not shuffle else None,
        shuffle=shuffle,
        collate_fn=collate_fn
    )
    
    return dataloader


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out)
        return out


input_size = 1
hidden_size = 20
output_size = 1
model = SimpleRNN(input_size, hidden_size, output_size)