from utils.config import *
from utils.preprocessing import load_all_data, handle_missing_values, handle_outliers, aggregate_data, normalize_df_minmax, split_data
from models.RNN import RNN  
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
    
    best_model, best_val_loss = None, -1
    train_loss_history, train_rmse_history = [], []
    val_loss_history, val_rmse_history = [], []

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss() 
    
    for epoch in range(epochs):
        model.to(device)
        model.train()
        epoch_loss = []
        
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
            epoch_loss.append(loss.item())
        
        train_loss = sum(epoch_loss) / len(epoch_loss)

        _, train_mse, train_rmse, train_mae, train_r2 = test(model, train_loader, criterion, device)
        val_loss, val_mse, val_rmse, val_mae, val_r2 = test(model, val_loader, criterion, device)
        if (epoch + 1) % log_per == 0:
            print(f"Epoch {epoch + 1} [Train]: loss {train_loss}, mse: {train_mse}, "
                  f"rmse: {train_rmse}, mae {train_mae}, r2: {train_r2}")
            print(f"Epoch {epoch + 1} [Val]: loss {val_loss}, mse: {val_mse}, "
                  f"rmse: {val_rmse}, mae {val_mae}, r2: {val_r2}")
        train_loss_history.append(train_mse)
        train_rmse_history.append(train_rmse)
        val_loss_history.append(val_mse)
        val_rmse_history.append(val_rmse)
    
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
    
    if plot_history:
        plt.plot(train_loss_history, label="Train MSE")
        plt.plot(val_loss_history, label="Val MSE")
        plt.legend()
        plt.show()
        plt.close()

        plt.plot(train_rmse_history, label="Train RMSE")
        plt.plot(val_rmse_history, label="Val RMSE")
        plt.legend()
        plt.show()
        plt.close()
        print(f"Best Validation Loss: {best_val_loss}")

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

def main():
    df_all = load_all_data(POBLESEC_ST)
    df_all = handle_missing_values(df_all)
    df_all = handle_outliers(df_all, 5, 95) 
    df_all = aggregate_data(df_all, 60)
    df_all, scaler = normalize_df_minmax(df_all, feature_range=(0, 1))
    df_train, df_val, df_test = split_data(df_all, 0.6, 0.2, 0.2) 

    print(df_train.shape)
    print(df_train.columns)


    train_loader = create_dataloader(df_train, target_columns=['down'], batch_size=32, seq_length=10)
    val_loader = create_dataloader(df_val, target_columns=['down'], batch_size=32, seq_length=10)
    test_loader = create_dataloader(df_test, target_columns=['down'], batch_size=32, seq_length=10)
    model = RNN(input_dim=len(df_train.columns), rnn_hidden_size=128, num_rnn_layers=2, rnn_dropout=0.2,
                layer_units=[128, 64], num_outputs=1, init_weights=True, matrix_rep=False, exogenous_dim=0)
    model = train(model, train_loader, val_loader, epochs=1, lr=0.001, max_grad_norm=5.0, plot_history=False)
       
    # test_loss, test_mse, test_rmse, test_mae, test_r2 = test(model, test_loader, nn.MSELoss(), device="cuda" if cuda.is_available() else "cpu")
    # print(f"Test Loss: {test_loss}, Test MSE: {test_mse}, Test RMSE: {test_rmse}, "
    #       f"Test MAE: {test_mae}, Test R2: {test_r2}")

if __name__ == "__main__":
    main()