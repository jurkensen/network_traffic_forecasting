import torch
import torch.nn as nn
import torch.cuda as cuda
from utils.preprocessing import *
from models.Dataloader import *
from typing import List
from utils.preprocessing import *
from models.train_utils import train, test_with_original_scale

class LSTM(torch.nn.Module):
    def __init__(self, input_dim: int, lstm_hidden_size: int = 128, num_lstm_layers: int = 2, 
                 lstm_dropout: float = 0.0, layer_units: List[int] = [128, 64], num_outputs: int = 1, 
                 init_weights: bool = True, exogenous_dim: int = 0):
        """
        LSTM model based on the research paper architecture.
        
        Args:
            input_dim: Number of input features
            lstm_hidden_size: Hidden size for LSTM layers (default 128 as per paper)
            num_lstm_layers: Number of LSTM layers (default 2 as per paper)
            lstm_dropout: Dropout rate for LSTM layers
            layer_units: Units for MLP layers after LSTM (default [128, 64] as per paper)
            num_outputs: Number of output predictions
            init_weights: Whether to initialize weights with Xavier uniform
            exogenous_dim: Dimension of exogenous features
        """
        super(LSTM, self).__init__()
        
        activation = torch.nn.ReLU()
        self.hidden_dim = lstm_hidden_size
        self.layer_dim = num_lstm_layers
        
        # following paper architecture with 128 and 64 units
        self.lstm = torch.nn.LSTM(
            input_size=input_dim, 
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers, 
            batch_first=True,
            dropout=lstm_dropout if num_lstm_layers > 1 else 0.0  # Dropout only if multiple layers
        )
        
        # MLP layers after LSTM
        if len(layer_units) == 1:
            layers = [torch.nn.Linear(lstm_hidden_size + exogenous_dim, num_outputs)]
        else:
            # Input layer
            layers = [torch.nn.Linear(lstm_hidden_size + exogenous_dim, layer_units[0]), activation]
            
            # Hidden layers
            for i in range(len(layer_units) - 1):
                layers.append(torch.nn.Linear(layer_units[i], layer_units[i + 1]))
                layers.append(activation)
            
            # Output layer
            layers.append(torch.nn.Linear(layer_units[len(layer_units) - 1], num_outputs))
        
        self.MLP_layers = torch.nn.Sequential(*layers)
        
        if init_weights:
            self.MLP_layers.apply(self._init_weights)
            # Initialize LSTM weights
            self._init_lstm_weights()
    
    def forward(self, x, exogenous_data=None, device="cpu"):
        """
        Forward pass through the LSTM model.
        
        Args:
            x: Input sequences [batch_size, seq_length, input_dim]
            exogenous_data: Optional exogenous features [batch_size, exogenous_dim]
            device: Device to run computation on
            
        Returns:
            out: Predictions [batch_size, num_outputs]
        """
        x = x.reshape(x.size(0), x.size(1), x.size(2))
        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        
        # LSTM forward pass
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        # Take the output from the last time step (as mentioned in paper)
        out = out[:, -1, :]  # [batch_size, hidden_dim]
        
        # Concatenate with exogenous data if provided
        if exogenous_data is not None and exogenous_data.numel() > 0:
            out = torch.cat((out, exogenous_data), dim=1)
        
        # Pass through MLP layers
        out = self.MLP_layers(out)
        
        return out
    
    def _init_weights(self, module):
        """Initialize weights for linear layers using Xavier uniform initialization."""
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def _init_lstm_weights(self):
        """Initialize LSTM weights."""
        for name, param in self.lstm.named_parameters():
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
    df_all = aggregate_data(df_all, 60)
    df_all, scaler = normalize_df_minmax(df_all, feature_range=(0, 1))
    df_train, df_val, df_test = split_data(df_all, 0.6, 0.2, 0.2) 
    original_columns = list(df_all.columns) 

    train_loader = create_dataloader(df_train, target_columns=['down'], batch_size=32, seq_length=12)
    val_loader = create_dataloader(df_val, target_columns=['down'], batch_size=32, seq_length=12)
    test_loader = create_dataloader(df_test, target_columns=['down'], batch_size=32, seq_length=12)
    
    # Use LSTM model instead of RNN - following the paper architecture
    # 128 units in first layer, 64 units in second layer, as described in the paper
    model = LSTM(
        input_dim=len(df_train.columns), 
        lstm_hidden_size=128,
        num_lstm_layers=2,     
        lstm_dropout=0.3,
        layer_units=[64], 
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