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
        )
        
        # MLP parameters
        self.fc1 = nn.Linear(self.rnn_hidden_size, 128)
        self.fc2 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()
        

    
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
                device='cpu'):
    """
    Train the model with gradient clipping and learning rate scheduling
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data (optional)
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cpu' or 'cuda')    
    Returns:
        Trained model and training history
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Training history
    history = {
        'train_loss': [],
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
                        
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
                    
        # Calculate average training loss
        avg_train_loss = train_loss / max(num_batches, 1)  # Avoid division by zero
        history['train_loss'].append(avg_train_loss)

        
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}')
    
    return model, history

def evaluate_model(model, train_data, test_loader, scaler, device='cpu'):
    """
    Evaluate the model on test data and average predictions for the same timestamp
   
    Args:
        model: Trained model
        train_data: Training data (for column information)
        test_loader: DataLoader for test data
        scaler: Scaler object used to normalize/denormalize data
        device: Device to evaluate on
   
    Returns:
        Dictionary of evaluation metrics
    """
    model = model.to(device)
    model.eval()
   
    # Dictionary to collect all predictions by timestamp
    predictions_by_timestamp = {}
    targets_by_timestamp = {}
   
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            targets = batch['target'].squeeze().cpu().numpy()
           
            # Get predictions
            outputs = model(features).squeeze().cpu().numpy()
           
            # Process timestamps and organize predictions by timestamp
            if isinstance(batch['target_timestamps'][0], list):
                # Multi-step forecasting
                for i, ts_list in enumerate(batch['target_timestamps']):
                    for j, ts in enumerate(ts_list):
                        # Convert timestamp to string for dictionary key
                        ts_key = str(ts)
                        
                        # Initialize lists if this timestamp hasn't been seen before
                        if ts_key not in predictions_by_timestamp:
                            predictions_by_timestamp[ts_key] = []
                            targets_by_timestamp[ts_key] = []
                            
                        # Add this prediction and target
                        if isinstance(outputs, np.ndarray) and outputs.ndim > 0 and i < len(outputs):
                            pred_val = outputs[i, j] if outputs.ndim > 1 else outputs[i]
                            predictions_by_timestamp[ts_key].append(pred_val)
                            
                            targ_val = targets[i, j] if targets.ndim > 1 else targets[i]
                            targets_by_timestamp[ts_key].append(targ_val)
            else:
                # Single-step forecasting
                for i in range(len(batch['target_timestamps'])):
                    ts = batch['target_timestamps'][i]
                    ts_key = str(ts)
                    
                    # Initialize lists if this timestamp hasn't been seen before
                    if ts_key not in predictions_by_timestamp:
                        predictions_by_timestamp[ts_key] = []
                        targets_by_timestamp[ts_key] = []
                    
                    # Add this prediction and target
                    if i < len(outputs):
                        pred_val = outputs[i]
                        predictions_by_timestamp[ts_key].append(pred_val)
                        
                        targ_val = targets[i] if i < len(targets) else None
                        if targ_val is not None:
                            targets_by_timestamp[ts_key].append(targ_val)
    
    # Average the predictions for each timestamp
    unique_timestamps = sorted(predictions_by_timestamp.keys())
    avg_predictions = []
    avg_targets = []
    timestamps = []
    
    for ts_key in unique_timestamps:
        # Average predictions for this timestamp
        if predictions_by_timestamp[ts_key]:
            avg_pred = np.mean(predictions_by_timestamp[ts_key])
            avg_predictions.append(avg_pred)
            
            # For targets, we should have the same value for each timestamp, but average to be safe
            avg_targ = np.mean(targets_by_timestamp[ts_key]) if targets_by_timestamp[ts_key] else None
            if avg_targ is not None:
                avg_targets.append(avg_targ)
                timestamps.append(pd.Timestamp(ts_key))
    
    print(f"Collected predictions for {len(unique_timestamps)} unique timestamps")
    print(f"Number of averaged predictions: {len(avg_predictions)}")
    print(f"Number of averaged targets: {len(avg_targets)}")
    
    # Convert to numpy arrays
    avg_predictions = np.array(avg_predictions)
    avg_targets = np.array(avg_targets)
    
    # Denormalize predictions and targets
    target_column_idx = train_data.columns.get_loc('down')  # Get column index of target
    
    # Create arrays filled with zeros (placeholder values)
    targets_denorm_array = np.zeros((len(avg_targets), len(train_data.columns)))
    preds_denorm_array = np.zeros((len(avg_predictions), len(train_data.columns)))
    
    # Set target values at the right column
    targets_denorm_array[:, target_column_idx] = avg_targets
    preds_denorm_array[:, target_column_idx] = avg_predictions
    
    # Inverse transform
    targets_denorm_array = scaler.inverse_transform(targets_denorm_array)
    preds_denorm_array = scaler.inverse_transform(preds_denorm_array)
    
    # Extract just the target column after denormalization
    denorm_targets = targets_denorm_array[:, target_column_idx]
    denorm_predictions = preds_denorm_array[:, target_column_idx]
    
    mse = mean_squared_error(denorm_targets, denorm_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(denorm_targets, denorm_predictions)
    mask = denorm_targets != 0
    mape = np.mean(np.abs((denorm_targets[mask] - denorm_predictions[mask]) / denorm_targets[mask])) * 100
    
    r2 = r2_score(denorm_targets, denorm_predictions)
    
    # Create results dictionary
    results = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'predictions': denorm_predictions,
        'targets': denorm_targets,
        'timestamps': timestamps
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
    
    indices = range(len(results['targets']))
    max_points = 100
    targets = results['targets']
    predictions = results['predictions']

    data_length = len(targets)
    if data_length > max_points:
        step = data_length // max_points
    else:
        step = 1
    
    # Downsample by taking every 'step' points
    indices = range(0, data_length, step)
    downsampled_targets = targets[::step]
    downsampled_predictions = predictions[::step]
    
    # Plot downsampled data
    plt.plot(indices, downsampled_targets, 'o-', label='Actual', markersize=4, alpha=0.7)
    plt.plot(indices, downsampled_predictions, 'x-', label='Predicted', markersize=4, alpha=0.7)

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
    plt.savefig('RNN_predictions_25_epochs.png')

def plot_loss_history(history):
    """
    Plot the training and validation loss history
    
    Args:
        history: Training history from train_model
    """
    plt.figure(figsize=(12, 6))

    
    # Plot loss
    plt.plot(history['train_loss'], label='Training Loss')
    plt.title('Loss Over Time')
    plt.ylabel('Loss')
    plt.xlabel('Epoch #')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    plt.savefig('loss_history_50_epochs.png')

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and prepare data
    train_data, test_data = load_data(POBLESEC_ST)
    train_data, test_data = aggregate_data(train_data), aggregate_data(test_data)
    train_data, test_data = preprocess_outliers(train_data), preprocess_outliers(test_data)
    normalized_train_data, scaler = normalize_df_minmax(train_data)

    normalized_test_data = pd.DataFrame(
        scaler.transform(test_data),
        index=test_data.index,
        columns=test_data.columns
    )
    
    # Create datasets
    train_dataset = NetworkDataset(normalized_train_data, 'down')
    test_dataset = NetworkDataset(normalized_test_data, 'down')
    
    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset, 
        batch_size=32,  
        seq_length=12,
        forecast_horizon=1
    )

    test_loader = create_dataloader(
        test_dataset, 
        batch_size=32,  
        seq_length=12,
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
        epochs=30,
        learning_rate=0.0003,  
        device=device,
    )
    
    # Plot training history
    plot_loss_history(history)
    

    # Evaluate model
    test_results = evaluate_model(trained_model, train_data, test_loader, scaler, device)
    
    # Print evaluation metrics
    print("\nTest Results:")
    print(f"MSE: {test_results['mse']:.4f}")
    print(f"RMSE: {test_results['rmse']:.4f}")
    print(f"MAE: {test_results['mae']:.4f}")
    print(f"MAPE: {test_results['mape']:.4f}")
    print(f"R²: {test_results['r2']:.4f}")
    
    # Plot predictions
    plot_predictions(test_results, title="RNN-MLP Predictions vs Actual Values")
    
    # Save model
    torch.save(trained_model.state_dict(), 'rnn_mlp_model.pth')
    print("Model saved to 'rnn_mlp_model.pth'")

if __name__ == "__main__":
    main()