import torch
import torch.nn as nn
import torch.cuda as cuda
from utils.preprocessing import *
from models.Dataloader import *
from typing import List
from utils.preprocessing import *
from models.train_utils import train, test_with_original_scale

class LSTM(torch.nn.Module):
    def __init__(self, input_dim: int, lstm_hidden_size1: int = 128, lstm_hidden_size2: int = 64, num_lstm_layers: int = 1, lstm_dropout: float = 0.0, num_outputs: int = 1, init_weights: bool = True, exogenous_dim: int = 0):
        """
        LSTM model with two LSTM layers: first with 128 units, second with 64 units, and a single output node.
        No FFNN/MLP layers after LSTM.
        """
        super(LSTM, self).__init__()
        self.hidden_dim1 = lstm_hidden_size1
        self.hidden_dim2 = lstm_hidden_size2
        self.layer_dim = num_lstm_layers
        self.exogenous_dim = exogenous_dim

        # First LSTM layer
        self.lstm1 = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_size1,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if num_lstm_layers > 1 else 0.0
        )
        # Second LSTM layer
        self.lstm2 = torch.nn.LSTM(
            input_size=lstm_hidden_size1,
            hidden_size=lstm_hidden_size2,
            num_layers=1,
            batch_first=True
        )
        # Output layer
        self.output_layer = torch.nn.Linear(lstm_hidden_size2 + exogenous_dim, num_outputs)

        if init_weights:
            self._init_weights(self.output_layer)
            self._init_lstm_weights(self.lstm1)
            self._init_lstm_weights(self.lstm2)

    def forward(self, x, exogenous_data=None, device="cpu"):
        x = x.reshape(x.size(0), x.size(1), x.size(2))
        h0_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim1).to(device)
        c0_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim1).to(device)
        out1, _ = self.lstm1(x, (h0_1, c0_1))

        h0_2 = torch.zeros(1, x.size(0), self.hidden_dim2).to(device)
        c0_2 = torch.zeros(1, x.size(0), self.hidden_dim2).to(device)
        out2, _ = self.lstm2(out1, (h0_2, c0_2))

        out = out2[:, -1, :]  # Take last time step
        if exogenous_data is not None and exogenous_data.numel() > 0:
            out = torch.cat((out, exogenous_data), dim=1)
        out = self.output_layer(out)
        return out

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def _init_lstm_weights(self, lstm):
        for name, param in lstm.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

def main():
    torch.manual_seed(42)
    df_all = load_all_data(POBLESEC_ST)
    df_all = handle_missing_values(df_all)
    df_all = handle_outliers(df_all, 5, 95) 
    df_all = aggregate_data(df_all, 2)
    df_all, scaler = normalize_df_minmax(df_all, feature_range=(0, 1))
    df_train, df_val, df_test = split_data(df_all, 0.6, 0.2, 0.2) 
    original_columns = list(df_all.columns) 

    train_loader = create_dataloader(df_train, target_columns=['down'], batch_size=32, seq_length=12)
    val_loader = create_dataloader(df_val, target_columns=['down'], batch_size=32, seq_length=12)
    test_loader = create_dataloader(df_test, target_columns=['down'], batch_size=32, seq_length=12)
    
    model = LSTM(
        input_dim=len(df_train.columns), 
        lstm_hidden_size1=128,
        lstm_hidden_size2=64,
        num_lstm_layers=1,     
        lstm_dropout=0.3,
        num_outputs=1,      
        init_weights=True, 
        exogenous_dim=0
    )
    
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train the model
    model = train(model, train_loader, val_loader, epochs=100, lr=0.0001, max_grad_norm=5.0, plot_history=True)
    
    # Test the model
    test_metrics = test_with_original_scale(model, test_loader, nn.MSELoss(), 
                                           device="cuda" if cuda.is_available() else "cpu",
                                           scaler=scaler, target_columns=['down'], 
                                           all_columns=original_columns)
    
    print("\n" + "="*50)
    print("FINAL TEST RESULTS - LSTM MODEL")
    print("="*50)
    print("Original Scale Metrics:")
    print(f"  RMSE: {test_metrics['original_scale']['rmse']:.2f}")
    print(f"  MAE:  {test_metrics['original_scale']['mae']:.2f}")
    print(f"  R²:   {test_metrics['original_scale']['r2']:.4f}")
    print(f"  MSE:  {test_metrics['original_scale']['mse']:.2f}")
    
    print("\nScaled Metrics (0-1 range):")
    print(f"  RMSE: {test_metrics['scaled']['rmse']:.6f}")
    print(f"  MAE:  {test_metrics['scaled']['mae']:.6f}")
    print(f"  R²:   {test_metrics['scaled']['r2']:.4f}")
    print(f"  Loss: {test_metrics['loss']:.6f}")

if __name__ == "__main__":
    main()