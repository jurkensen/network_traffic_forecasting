import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.config import *
from utils.preprocessing import *
from models.Dataloader import *

class RNN_MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(RNN_MLP, self).__init__()

        # RNN parameters
        self.rnn_hidden_size = 128
        self.rnn = nn.RNN(
            input_size=input_size, 
            hidden_size=self.rnn_hidden_size, 
            num_layers=1, 
            batch_first=True,
            nonlinearity='tanh'  # Using tanh for better gradient stability
        )
        
        # MLP parameters
        self.fc1 = nn.Linear(self.rnn_hidden_size, 128)
        self.fc2 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()
        
        # Initialize weights for better stability
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Initialize RNN weights with a small uniform distribution
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
                
        # Initialize linear layers
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        # Check for NaN in input
        if torch.isnan(x).any():
            print("NaN detected in the input data!")
            # Replace NaNs with zeros
            x = torch.nan_to_num(x, nan=0.0)
        
        # x shape: [batch_size, seq_length, n_features]
        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, self.rnn_hidden_size).to(x.device)
        
        # Process through RNN
        rnn_out, _ = self.rnn(x, h0)
        
        # Extract the output of the last timestep
        last_out = rnn_out[:, -1, :]
        
        # Feed through MLP
        x = self.relu(self.fc1(last_out))
        x = self.fc2(x)
        
        return x

def check_data_validity(dataloader):
    """Check for NaN or infinite values in the dataset"""
    for batch in dataloader:
        features = batch['features']
        targets = batch['target']
        
        if torch.isnan(features).any() or torch.isinf(features).any():
            print("Warning: NaN or Inf detected in features!")
            print(f"NaN count: {torch.isnan(features).sum()}")
            print(f"Inf count: {torch.isinf(features).sum()}")
            
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            print("Warning: NaN or Inf detected in targets!")
            print(f"NaN count: {torch.isnan(targets).sum()}")
            print(f"Inf count: {torch.isinf(targets).sum()}")
    
    print("Data validity check completed.")

def train_model(model, train_loader, val_loader=None, epochs=25, learning_rate=0.0005, 
                device='cpu', grad_clip_value=1.0):
    """
    Train the model with gradient clipping and learning rate scheduling
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data (optional)
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cpu' or 'cuda')
        grad_clip_value: Maximum gradient norm for gradient clipping
    
    Returns:
        Trained model and training history
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }
    
    print(f"Training on {device}...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            features = batch['features'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            # Check for NaN loss
            if torch.isnan(loss).any():
                print(f"NaN loss detected at batch {batch_idx}, epoch {epoch+1}")
                continue  # Skip this batch
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
            
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            # Print batch progress every 20 batches
            if (batch_idx + 1) % 20 == 0:
                print(f'  Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.6f}')
        
        # Calculate average training loss
        avg_train_loss = train_loss / max(num_batches, 1)  # Avoid division by zero
        history['train_loss'].append(avg_train_loss)
        
        # Update learning rate based on training loss
        scheduler.step(avg_train_loss)
        
        # Store current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
        
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, LR: {current_lr:.6f}')
    
    return model, history

def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluate the model on test data
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to evaluate on
    
    Returns:
        Dictionary of evaluation metrics
    """
    model = model.to(device)
    model.eval()
    
    all_targets = []
    all_predictions = []
    all_timestamps = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            targets = batch['target'].squeeze().cpu().numpy()
            
            # Get predictions
            outputs = model(features).squeeze().cpu().numpy()
            
            # Store predictions and targets
            all_predictions.extend(outputs)
            all_targets.extend(targets)
            
            # Store timestamps for plotting
            if isinstance(batch['target_timestamps'][0], list):
                # Multi-step forecasting
                for ts_list in batch['target_timestamps']:
                    all_timestamps.extend(ts_list)
            else:
                # Single-step forecasting
                all_timestamps.extend(batch['target_timestamps'])
    
    # Convert to numpy arrays
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    
    # Calculate error metrics
    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    
    # Create results dictionary
    results = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': all_predictions,
        'targets': all_targets,
        'timestamps': all_timestamps
    }
    
    return results

def plot_predictions(results, title="Model Predictions vs Actual Values"):
    """
    Plot the predictions against actual values
    
    Args:
        results: Results dictionary from evaluate_model
        title: Plot title
    """
    plt.figure(figsize=(12, 6))
    
    # If timestamps are datetime objects, convert to matplotlib dates
    timestamps = results['timestamps']
    
    # Plot actual vs predicted
    plt.plot(timestamps, results['targets'], label='Actual', marker='o', linestyle='-', markersize=4)
    plt.plot(timestamps, results['predictions'], label='Predicted', marker='x', linestyle='-', markersize=4)
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # Add metrics as text
    metrics_text = (
        f"MSE: {results['mse']:.4f}\n"
        f"RMSE: {results['rmse']:.4f}\n"
        f"MAE: {results['mae']:.4f}\n"
        f"R²: {results['r2']:.4f}"
    )
    
    plt.figtext(0.02, 0.02, metrics_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def plot_loss_history(history):
    """
    Plot the training and validation loss history
    
    Args:
        history: Training history from train_model
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.set_title('Loss Over Time')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot learning rate
    if 'learning_rates' in history:
        ax2.plot(history['learning_rates'], color='red')
        ax2.set_title('Learning Rate')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and prepare data
    train_data, test_data = load_data(POBLESEC_ST)
    
    # Check for NaN values in data
    print("Checking training data for NaN values...")
    print(f"NaNs in train data: {train_data.isna().sum().sum()}")
    print(f"NaNs in test data: {test_data.isna().sum().sum()}")
    
    # Fill NaN values if present
    if train_data.isna().sum().sum() > 0 or test_data.isna().sum().sum() > 0:
        print("Filling NaN values with forward fill, then backward fill")
        train_data = train_data.fillna(method='ffill').fillna(method='bfill')
        test_data = test_data.fillna(method='ffill').fillna(method='bfill')
    
    # Create datasets
    train_dataset = NetworkDataset(train_data, 'down')
    test_dataset = NetworkDataset(test_data, 'down')
    
    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset, 
        batch_size=32,  
        seq_length=6,
        forecast_horizon=1
    )
    
    test_loader = create_dataloader(
        test_dataset, 
        batch_size=32,  
        seq_length=6,
        forecast_horizon=1
    )
    
    # Check data validity
    print("Checking data validity...")
    check_data_validity(train_loader)
    
    # Create model
    input_size = len(train_data.columns)  # Number of features
    output_size = 1  # Predicting a single value
    model = RNN_MLP(input_size, output_size)
    
    # Print model summary
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    
    # Train model
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=None,
        epochs=50,
        learning_rate=0.0003,  # Reduced learning rate
        device=device,
        grad_clip_value=1.0  # Add gradient clipping
    )
    
    # Plot training history
    plot_loss_history(history)
    
    # Evaluate model
    test_results = evaluate_model(trained_model, test_loader, device)
    
    # Print evaluation metrics
    print("\nTest Results:")
    print(f"MSE: {test_results['mse']:.4f}")
    print(f"RMSE: {test_results['rmse']:.4f}")
    print(f"MAE: {test_results['mae']:.4f}")
    print(f"R²: {test_results['r2']:.4f}")
    
    # Plot predictions
    plot_predictions(test_results, title="RNN-MLP Predictions vs Actual Values")
    
    # Save model
    torch.save(trained_model.state_dict(), 'rnn_mlp_model.pth')
    print("Model saved to 'rnn_mlp_model.pth'")

if __name__ == "__main__":
    main()