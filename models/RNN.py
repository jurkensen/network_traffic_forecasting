import torch
from utils.config import *
from utils.preprocessing import *
from models.Dataloader import *
from typing import List
from models.train_utils import *

class RNN(torch.nn.Module):
    def __init__(self, input_dim: int, rnn_hidden_size: int = 128, num_rnn_layers: int = 2, rnn_dropout: float = 0.0,
                 layer_units: List[int] = [128, 64], num_outputs: int = 2, init_weights: bool = True, exogenous_dim: int = 0):
        super(RNN, self).__init__()

        activation = torch.nn.ReLU()

        self.hidden_dim = rnn_hidden_size
        self.layer_dim = num_rnn_layers

        self.rnn = torch.nn.RNN(input_size=input_dim, hidden_size=rnn_hidden_size,
                                num_layers=num_rnn_layers, batch_first=True,
                                dropout=rnn_dropout)

        if len(layer_units) == 1:
            layers = [torch.nn.Linear(rnn_hidden_size + exogenous_dim, num_outputs)]
        else:
            layers = [torch.nn.Linear(rnn_hidden_size + exogenous_dim, layer_units[0]), activation]  # Input layer
            for i in range(len(layer_units) - 1):  # hidden layers
                layers.append(torch.nn.Linear(layer_units[i], layer_units[i + 1]))
                layers.append(activation)
            layers.append(torch.nn.Linear(layer_units[len(layer_units) - 1], num_outputs))  # Output layer

        self.MLP_layers = torch.nn.Sequential(*layers)
        if init_weights:
            self.MLP_layers.apply(self._init_weights)

    def forward(self, x, exogenous_data=None, device="cpu"):
        x = x.reshape(x.size(0), x.size(1), x.size(2))

        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)

        out, _ = self.rnn(x, h0.detach())
        out = out[:, -1, :]

        if exogenous_data is not None:
            out = torch.cat((out, exogenous_data), dim=1)

        out = self.MLP_layers(out)
        return out

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()


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
    model = RNN(input_dim=len(df_train.columns), rnn_hidden_size=128, num_rnn_layers=2, rnn_dropout=0.3,
                layer_units=[128, 64], num_outputs=1, init_weights=True, exogenous_dim=0)
    print(model)
    model = train(model, train_loader, val_loader, epochs=100, lr=0.0001, max_grad_norm=5.0, plot_history=True)
    test_metrics = test_with_original_scale(model, test_loader, nn.MSELoss(), 
                                           device="cuda" if cuda.is_available() else "cpu",
                                           scaler=scaler, target_columns=['down'], 
                                           all_columns=original_columns)
    
    print("\n" + "="*50)
    print("FINAL TEST RESULTS")
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