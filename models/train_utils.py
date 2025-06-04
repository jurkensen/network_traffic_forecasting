from utils.config import *
from utils.preprocessing import load_all_data, handle_missing_values, handle_outliers, aggregate_data, normalize_df_minmax, split_data
from models.RNN import RNN  
import utils.plot as plot_utils
from models.Dataloader import create_dataloader

import torch
import torch.nn as nn
import torch.cuda as cuda
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch.optim as optim

import math
import torch
import copy
import numpy as np
from typing import Tuple, List, Union, Dict
from matplotlib import pyplot as plt



def train(model: torch.nn.Module,
          train_loader,
          val_loader,
          epochs: int = 10,
          lr: float = 0.001,
          max_grad_norm: float = 0.,
          plot_history: bool = True,
          device="cuda" if cuda.is_available() else "cpu",
          log_per: int = 1):
    """Trains a neural network defined as torch module."""
    
    best_model, best_val_loss = None, float('inf')
    train_loss_history, train_rmse_history = [], []
    val_loss_history, val_rmse_history = [], []

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss() 
    
    for epoch in range(epochs):
        model.to(device)
        model.train()
        
        for x, exogenous, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            if exogenous is not None and len(exogenous) > 0:
                exogenous = exogenous.to(device)
            else:
                exogenous = None

            optimizer.zero_grad()
            y_pred = model(x, exogenous, device)
            loss = criterion(y_pred, y)
            loss.backward()
            if max_grad_norm > 0.:
                clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        
        train_loss, train_mse, train_rmse, train_mae, train_r2 = test(model, train_loader, criterion, device)
        val_loss, val_mse, val_rmse, val_mae, val_r2 = test(model, val_loader, criterion, device)
        if (epoch + 1) % log_per == 0:
            print(f"Epoch {epoch + 1} [Train]: loss {train_loss}, mse: {train_mse}, "
                  f"rmse: {train_rmse}, mae {train_mae}, r2: {train_r2}")
            print(f"Epoch {epoch + 1} [Val]: loss {val_loss}, mse: {val_mse}, "
                  f"rmse: {val_rmse}, mae {val_mae}, r2: {val_r2}")
        train_loss_history.append(train_loss)
        train_rmse_history.append(train_rmse)
        val_loss_history.append(val_loss)
        val_rmse_history.append(val_rmse)
    
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
    
    if plot_history:
        plt.plot(train_loss_history, label="Train MSE")
        plt.plot(val_loss_history, label="Val MSE")
        plt.title("Train and Validation Loss")
        plt.legend()
        plt.show()
        plt.savefig("train_val_loss.png")
        plt.close()

    return best_model


def test(model, data, criterion, device="cuda") -> Union[
    Tuple[float, float, float, float], List[torch.tensor], torch.tensor]:
    """Tests a trained model."""
    model.to(device)
    model.eval()
    y_true, y_pred = [], []
    loss = 0.
    with torch.no_grad():
        for x, exogenous, y in data:
            x, y = x.to(device), y.to(device)

            if exogenous is not None and len(exogenous) > 0:
                exogenous = exogenous.to(device)
            else:
                exogenous = None

            out = model(x, exogenous, device)

            if criterion is not None:
                loss += criterion(out, y).item()

            y_true.extend(y)
            y_pred.extend(out)

    loss /= len(data.dataset)

    y_true = torch.stack(y_true)
    y_pred = torch.stack(y_pred)
    mse, rmse, mae, r2 = accumulate_metric(y_true.cpu(), y_pred.cpu())

    return loss, mse, rmse, mae, r2

def accumulate_metric(y_true, y_pred):
    if not isinstance(y_true, np.ndarray):
        y_true = y_true.cpu().numpy()
    if not isinstance(y_pred, np.ndarray):
        y_pred = y_pred.cpu().numpy()
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return mse, rmse, mae, r2
def accumulate_metric_original_scale(y_true_scaled, y_pred_scaled, scaler, target_columns, all_columns):
    """Calculate metrics in original scale"""
    
    # Convert to numpy if needed
    if not isinstance(y_true_scaled, np.ndarray):
        y_true_scaled = y_true_scaled.cpu().numpy()
    if not isinstance(y_pred_scaled, np.ndarray):
        y_pred_scaled = y_pred_scaled.cpu().numpy()
    
    # Inverse transform to original scale
    y_true_original = inverse_transform_targets(y_true_scaled, scaler, target_columns, all_columns)
    y_pred_original = inverse_transform_targets(y_pred_scaled, scaler, target_columns, all_columns)
    
    # Calculate metrics in original scale
    mse = mean_squared_error(y_true_original, y_pred_original)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true_original, y_pred_original)
    r2 = r2_score(y_true_original, y_pred_original)
    
    # Also calculate scaled metrics for comparison
    mse_scaled = mean_squared_error(y_true_scaled, y_pred_scaled)
    rmse_scaled = math.sqrt(mse_scaled)
    mae_scaled = mean_absolute_error(y_true_scaled, y_pred_scaled)
    r2_scaled = r2_score(y_true_scaled, y_pred_scaled)
    
    
    plt.figure(figsize=(15, 7))
    plt.plot(y_true_original, label='Actual')
    plt.plot(y_pred_original, label='Forecast', color='red')
    plt.title(f'Forecast vs Actual for {target_columns[0].capitalize()} Traffic')
    plt.xlabel('Time Steps')
    plt.ylabel('Bitrate')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'model_forecast_{target_columns}.png')
    plt.close()

    return {
        'original_scale': {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        },
        'scaled': {
            'mse': mse_scaled,
            'rmse': rmse_scaled,
            'mae': mae_scaled,
            'r2': r2_scaled
        }
    }
def test_with_original_scale(model, data, criterion, device, scaler, target_columns, all_columns):
    """Test model and return metrics in both original and scaled space"""
    model.to(device)
    model.eval()
    y_true, y_pred = [], []
    loss = 0.
    
    with torch.no_grad():
        for x, exogenous, y in data:
            x, y = x.to(device), y.to(device)

            if exogenous is not None and len(exogenous) > 0:
                exogenous = exogenous.to(device)
            else:
                exogenous = None

            out = model(x, exogenous, device)

            if criterion is not None:
                loss += criterion(out, y).item()

            y_true.extend(y.cpu())
            y_pred.extend(out.cpu())

    loss /= len(data.dataset)

    y_true = torch.stack(y_true).numpy()
    y_pred = torch.stack(y_pred).numpy()
    
    # Calculate metrics in both scales
    metrics = accumulate_metric_original_scale(y_true, y_pred, scaler, target_columns, all_columns)
    metrics['loss'] = loss
    
    return metrics

def inverse_transform_targets(y_scaled, scaler, target_columns, all_columns):
    """
    Inverse transform only the target columns back to original scale
    
    Args:
        y_scaled: Scaled target values (numpy array)
        scaler: The fitted MinMaxScaler used for normalization
        target_columns: List of target column names
        all_columns: List of all column names in original order
    
    Returns:
        y_original: Targets in original scale
    """
    # Get indices of target columns
    target_indices = [all_columns.index(col) for col in target_columns]
    
    # Create a dummy array with all features (we only care about target columns)
    if len(y_scaled.shape) == 1:
        # Single target case
        dummy_array = np.zeros((len(y_scaled), len(all_columns)))
        dummy_array[:, target_indices[0]] = y_scaled
    else:
        # Multiple targets case
        dummy_array = np.zeros((y_scaled.shape[0], len(all_columns)))
        for i, idx in enumerate(target_indices):
            dummy_array[:, idx] = y_scaled[:, i]
    
    # Inverse transform the entire array
    original_scale = scaler.inverse_transform(dummy_array)
    
    # Extract only the target columns
    if len(target_columns) == 1:
        return original_scale[:, target_indices[0]]
    else:
        return original_scale[:, target_indices]
    