import torch
from utils.config import *
from utils.preprocessing import *
from models.Dataloader import *
from models.LSTM import get_predictions_for_plot
from typing import List
from models.train_utils import *
import os 
import json
from datetime import datetime
import matplotlib.pyplot as plt


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
    
    # Create results directory
    results_dir = f"rnn_interval_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data once
    df_original = load_all_data(POBLESEC_ST)
    df_original = handle_missing_values(df_original)
    df_original = handle_outliers(df_original, 5, 95)
    
    # Dictionary to store all results
    all_results = {}
    
    # Test different intervals
    intervals = [5, 15, 30, 60]
    
    # Option 1: Fixed temporal context (e.g., 6 hours)
    # CONTEXT_HOURS = 6
    
    # Option 2: Fixed sequence length
    FIXED_SEQ_LENGTH = 12
    
    for interval_minutes in intervals:
        print(f"\n{'='*60}")
        print(f"TESTING INTERVAL: {interval_minutes} MINUTES")
        print(f"{'='*60}")
        
        # Create interval-specific directory
        interval_dir = os.path.join(results_dir, f"interval_{interval_minutes}min")
        os.makedirs(interval_dir, exist_ok=True)
        
        # Aggregate data for this specific interval (fresh copy each time!)
        df_agg = aggregate_data(df_original.copy(), interval_minutes)
        df_agg = add_hour_features(df_agg, interval_minutes)
        exogenous_columns = ['hour', 'hour_sin', 'hour_cos']

        df_agg, scaler = normalize_df_minmax(df_agg, feature_range=(0, 1))
        
        # Split data
        df_train, df_val, df_test = split_data(df_agg, 0.6, 0.2, 0.2)
        original_columns = list(df_agg.columns)
        
        # Calculate sequence length based on desired context
        # Option 1: Dynamic sequence length for fixed temporal context
        # seq_length = int(CONTEXT_HOURS * 60 / interval_minutes)
        
        # Option 2: Fixed sequence length (your current approach)
        seq_length = FIXED_SEQ_LENGTH
        
        # Option 3: Adaptive sequence length with minimum context
        seq_length = max(int(360 / interval_minutes), 6)  # At least 6 hours or 6 steps
        
        print(f"Using sequence length: {seq_length} (context: {seq_length * interval_minutes / 60:.1f} hours)")
        
        # Create dataloaders
        exog_train = df_train[exogenous_columns]
        exog_val = df_val[exogenous_columns]
        exog_test = df_test[exogenous_columns]
        
        # Create dataloaders with exogenous data
        train_loader = create_dataloader(
            dataframe=df_train,
            target_columns=['down'],
            batch_size=32,
            seq_length=seq_length,
            exogenous_data=exog_train
        )
        
        val_loader = create_dataloader(
            dataframe=df_val,
            target_columns=['down'],
            batch_size=32,
            seq_length=seq_length,
            exogenous_data=exog_val
        )
        
        test_loader = create_dataloader(
            dataframe=df_test,
            target_columns=['down'],
            batch_size=32,
            seq_length=seq_length,
            exogenous_data=exog_test
        )
        
        # Create RNN model with exogenous dimension
        model = RNN(
            input_dim=len(df_train.columns),  # Only main features, not exogenous
            rnn_hidden_size=128,
            num_rnn_layers=2,
            rnn_dropout=0.3,
            layer_units=[128, 64],
            num_outputs=1,
            init_weights=True,
            exogenous_dim=len(exogenous_columns)  # 3 in this case
        )
        
        print(f"\nModel architecture:")
        print(f"Input features: {len(df_train.columns)}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train the model
        model = train(model, train_loader, val_loader, epochs=50, lr=0.0001, 
                     max_grad_norm=5.0, plot_history=True, 
                     save_path=os.path.join(interval_dir, "training_history.png"))
        
        # Save model
        torch.save(model.state_dict(), 
                  os.path.join(interval_dir, f"model_interval_{interval_minutes}.pth"))
        
        # Test the model
        test_metrics = test_with_original_scale(
            model, test_loader, nn.MSELoss(),
            device="cuda" if torch.cuda.is_available() else "cpu",
            scaler=scaler, target_columns=['down'],
            all_columns=original_columns
        )
        
        # Generate predictions for visualization
        predictions, actuals = get_predictions_for_plot(
            model, test_loader, scaler, target_columns=['down'],
            all_columns=original_columns,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        plt.figure(figsize=(15, 6))
        plt.plot(actuals, label='Actual', color='blue', alpha=0.7)
        plt.plot(predictions, label='Predicted', color='red', alpha=0.7)
        plt.title(f'LSTM Predictions - {interval_minutes} min intervals\n'
                 f'RMSE: {test_metrics["rmse"]:.2f}, '
                 f'MAE: {test_metrics["mae"]:.2f}, '
                 f'R²: {test_metrics["r2"]:.4f}')
        plt.xlabel('Time Steps')
        plt.ylabel('Traffic (bytes)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(interval_dir, "predictions.png"), dpi=300)
        plt.close()
        
        # Store results
        results = {
            'interval_minutes': interval_minutes,
            'sequence_length': seq_length,
            'context_hours': seq_length * interval_minutes / 60,
            'data_shape': {
                'train': df_train.shape,
                'val': df_val.shape,
                'test': df_test.shape
            },
            'model_params': sum(p.numel() for p in model.parameters()),
            'metrics': test_metrics
        }
        
        all_results[interval_minutes] = results
        
        # Save individual results
        with open(os.path.join(interval_dir, "results.json"), 'w') as f:
            json.dump(results, f, indent=4)
        
        # Print results
        print(f"\n{'='*50}")
        print(f"RESULTS FOR {interval_minutes} MINUTE INTERVALS")
        print(f"{'='*50}")
        print(f"Sequence Length: {seq_length} (context: {seq_length * interval_minutes / 60:.1f} hours)")
        print("\nOriginal Scale Metrics:")
        print(f"  RMSE: {test_metrics['rmse']:.2f}")
        print(f"  MAE:  {test_metrics['mae']:.2f}")
        print(f" MAPE:  {test_metrics['mape']:.2f}")
        print(f"  R²:   {test_metrics['r2']:.4f}")
        print(f"  MSE:  {test_metrics['mse']:.2f}")
    
    # Save summary of all results
    with open(os.path.join(results_dir, "all_results_summary.json"), 'w') as f:
        json.dump(all_results, f, indent=4)



if __name__ == "__main__":
    main()