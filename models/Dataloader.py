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
        self.data = dataframe.values        
        self.target_col = target_column
        self.target_idx = dataframe.columns.get_loc(target_column)
        self.timestamps = list(dataframe.index)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row_data = self.data[idx]
        features = row_data.astype(np.float32)
        target = row_data[self.target_idx]
        timestamp = self.timestamps[idx]
       
        return {
            'features': torch.tensor(features),
            'target': torch.tensor(target, dtype=torch.float32),
            'timestamp': timestamp,
        }

class TimeSeriesBatchSampler:
    def __init__(self, data_source, batch_size, stride=1):
        self.data_source = data_source
        self.batch_size = batch_size
        self.stride = stride
        
    def __iter__(self):
        for i in range(0, len(self.data_source) - self.batch_size + 1, self.stride):
            yield list(range(i, i + self.batch_size))
    
    def __len__(self):
        return (len(self.data_source) - self.batch_size) // self.stride + 1

def time_series_collate_fn(batch, seq_length, forecast_horizon):
    """
    Custom collate function that forms sequences from individual timesteps
    
    Args:
        batch: List of samples from the dataset
        seq_length: Length of input sequences
        forecast_horizon: How many steps ahead to predict
    """
    # Check if we can form a valid sequence
    if len(batch) < seq_length + forecast_horizon:
        raise ValueError(f"Batch size ({len(batch)}) is smaller than required sequence length + forecast horizon ({seq_length + forecast_horizon})")
    
    # Prepare lists for sequences
    feature_seqs = []
    target_seqs = []
    timestamp_seqs = []
    target_timestamps = []
    
    # Create sliding window sequences
    for i in range(0, len(batch) - seq_length - forecast_horizon + 1):
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

def create_dataloader(dataset, batch_size=32, seq_length=6, forecast_horizon=1, shuffle=False):
    """
    Create DataLoader with time series support
    
    Args:
        dataset: Dataset to load from
        batch_size: Number of sequences per batch
        seq_length: Length of input sequences
        forecast_horizon: How many steps ahead to predict
        shuffle: Whether to shuffle the data
    """
    # Define custom collate function with closure to set parameters
    collate_fn = lambda batch: time_series_collate_fn(
        batch, 
        seq_length=seq_length,
        forecast_horizon=forecast_horizon,
    )

    # Calculate the adjusted batch size
    adjusted_batch_size = batch_size + seq_length + forecast_horizon - 1
    
    # Use custom sampler to preserve time ordering and enable overlapping batches
    batch_sampler = TimeSeriesBatchSampler(
        dataset, 
        batch_size=adjusted_batch_size, 
        stride=batch_size
    )    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
    )
    
    return dataloader

def main():
    train_poblesec, test_poblesec = load_data(POBLESEC_ST)
    train_dataset = NetworkDataset(train_poblesec, 'down')

    print(f"Dataset has {len(train_dataset)} entries")

    train_loader = create_dataloader(
            train_dataset, 
            batch_size=4,  
            seq_length=6,
            forecast_horizon=1
        )
    
    print(f"Dataset has {len(train_loader)} batches")
    
    # Get the first batch
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx > 1:  # Just look at the first couple of batches
            break
            
        print(f"\n--- Batch {batch_idx+1} ---")
        
        # Print batch information
        features = batch['features']
        targets = batch['target']
        timestamps = batch['timestamps']
        target_timestamps = batch['target_timestamps']
        
        print(f"Batch features shape: {features.shape}")
        print(f"Batch targets shape: {targets.shape}")
        
        # Print details of first sequence in batch
        print("\nFirst sequence in batch:")
        print(f"  Input timestamps: {timestamps[0]}")
        print(f"  Target timestamp: {target_timestamps[0]}")
        
        # Print feature values for first sequence
        print("\nFeature values for first sequence:")
        first_seq = features[0]  # Shape: [seq_length, num_features]
        
        # Get feature names from original dataframe
        feature_names = list(train_poblesec.columns)
        
        # Print first and last timestep features
        print("\nFirst timestep features:")
        for i, name in enumerate(feature_names):
            print(f"  {name}: {first_seq[0][i].item():.4f}")
            
        print("\nLast timestep features:")
        for i, name in enumerate(feature_names):
            print(f"  {name}: {first_seq[-1][i].item():.4f}")
            
        print("\nTarget value to predict:", targets[0].item())
        
        # Show the sliding window pattern
        if batch_idx == 0:
            print("\nSliding window pattern:")
            for i in range(len(batch['timestamps'])):
                print(f"Sequence {i+1}:")
                print(f"  Input: {batch['timestamps'][i]}")
                print(f"  Target: {batch['target_timestamps'][i]}")
                print()
main()