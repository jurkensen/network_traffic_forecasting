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
        self.feature_cols = list(dataframe.columns)
        self.timestamps = list(dataframe.index)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row_data = self.data[idx]
        features = row_data.astype(np.float32)
        timestamp = self.timestamps[idx]
       
        return {
            'features': torch.tensor(features),
            'timestamp': timestamp,
        }
    
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

def create_dataloader(dataframe, target_column, batch_size=32, seq_length=6, forecast_horizon=1, shuffle=False):
    # Create dataset
    dataset = NetworkDataset(dataframe, target_column)
    
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

def main():
    train_poblesec, test_poblesec = load_data(POBLESEC_ST)
    simple_dataset = NetworkDataset(train_poblesec, 'down')
    
    # Print dataset length
    print(f"Dataset length: {len(simple_dataset)}")
    
    # Print all parameters from the dataset object
    print("\nDataset parameters:")
    print(f"Target column: {simple_dataset.target_col}")
    print(f"Feature columns: {simple_dataset.feature_cols}")
    print(f"Data shape: {simple_dataset.data.shape}")
    print(f"Number of timestamps: {len(simple_dataset.timestamps)}")
    
    # Print first few timestamps and data
    print("\nFirst 3 timestamps:")
    for i in range(min(3, len(simple_dataset.timestamps))):
        print(f"  {simple_dataset.timestamps[i]}")
    
    # Display a sample
    first_sample = simple_dataset[0]
    print("\nSample from position 0:")
    print(f"Features shape: {first_sample['features'].shape}")
    print(f"Timestamp: {first_sample['timestamp']}")
    
    # Print feature names and values for the first sample
    print("\nFeature values for first sample:")
    for i, feature_name in enumerate(simple_dataset.feature_cols):
        print(f"  {feature_name}: {first_sample['features'][i].item()}")    # train_loader = create_dataloader(
    #         train_poblesec, 
    #         batch_size=4,  # Small batch size for inspection
    #         seq_length=6,
    #         forecast_horizon=1,
    #         target_column='down'
    #     )
    
    # print(f"Dataset has {len(train_loader)} batches")
    
    # # Get the first batch
    # for batch_idx, batch in enumerate(train_loader):
    #     if batch_idx > 1:  # Just look at the first couple of batches
    #         break
            
    #     print(f"\n--- Batch {batch_idx+1} ---")
        
    #     # Print batch information
    #     features = batch['features']
    #     targets = batch['target']
    #     timestamps = batch['timestamps']
    #     target_timestamps = batch['target_timestamps']
        
    #     print(f"Batch features shape: {features.shape}")
    #     print(f"Batch targets shape: {targets.shape}")
        
    #     # Print details of first sequence in batch
    #     print("\nFirst sequence in batch:")
    #     print(f"  Input timestamps: {timestamps[0]}")
    #     print(f"  Target timestamp: {target_timestamps[0]}")
        
    #     # Print feature values for first sequence
    #     print("\nFeature values for first sequence:")
    #     first_seq = features[0]  # Shape: [seq_length, num_features]
        
    #     # Get feature names from original dataframe
    #     feature_names = list(train_poblesec.columns)
        
    #     # Print first and last timestep features
    #     print("\nFirst timestep features:")
    #     for i, name in enumerate(feature_names):
    #         print(f"  {name}: {first_seq[0][i].item():.4f}")
            
    #     print("\nLast timestep features:")
    #     for i, name in enumerate(feature_names):
    #         print(f"  {name}: {first_seq[-1][i].item():.4f}")
            
    #     print("\nTarget value to predict:", targets[0].item())
        
    #     # Show the sliding window pattern
    #     if batch_idx == 0:
    #         print("\nSliding window pattern:")
    #         for i in range(min(3, len(batch['timestamps']))):
    #             print(f"Sequence {i+1}:")
    #             print(f"  Input: {batch['timestamps'][i]}")
    #             print(f"  Target: {batch['target_timestamps'][i]}")
    #             print()
main()