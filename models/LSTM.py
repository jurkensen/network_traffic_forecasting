import torch
import torch.nn as nn
import torch.cuda as cuda
from utils.preprocessing import *
from models.Dataloader import *
from typing import List
from utils.preprocessing import *
from models.train_utils import train, test_with_original_scale, inverse_transform_targets
import os
import json
from datetime import datetime


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
    
    # Create results directory
    results_dir = f"lstm_interval_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
        # seq_length = FIXED_SEQ_LENGTH
        
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
        
        # Create LSTM model with exogenous dimension
        model = LSTM(
            input_dim=len(df_train.columns),  # All columns including exogenous
            lstm_hidden_size1=128,
            lstm_hidden_size2=64,
            num_lstm_layers=1,
            lstm_dropout=0.3,
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
        
        # Create and save prediction plot
        plt.figure(figsize=(15, 6))
        plt.plot(actuals, label='Actual', color='blue', alpha=0.7)
        plt.plot(predictions, label='Predicted', color='red', alpha=0.7)
        plt.title(f'LSTM Predictions - {interval_minutes} min intervals\n'
                 f'RMSE: {test_metrics["original_scale"]["rmse"]:.2f}, '
                 f'MAE: {test_metrics["original_scale"]["mae"]:.2f}, '
                 f'R²: {test_metrics["original_scale"]["r2"]:.4f}')
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
            'metrics': {
                'original_scale': test_metrics['original_scale'],
                'scaled': test_metrics['scaled'],
                'loss': float(test_metrics['loss'])
            }
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
        print(f"  RMSE: {test_metrics['original_scale']['rmse']:.2f}")
        print(f"  MAE:  {test_metrics['original_scale']['mae']:.2f}")
        print(f"  R²:   {test_metrics['original_scale']['r2']:.4f}")
        print(f"  MSE:  {test_metrics['original_scale']['mse']:.2f}")
        print("\nScaled Metrics (0-1 range):")
        print(f"  RMSE: {test_metrics['scaled']['rmse']:.6f}")
        print(f"  MAE:  {test_metrics['scaled']['mae']:.6f}")
        print(f"  R²:   {test_metrics['scaled']['r2']:.4f}")
        print(f"  Loss: {test_metrics['loss']:.6f}")
    
    # Save summary of all results
    with open(os.path.join(results_dir, "all_results_summary.json"), 'w') as f:
        json.dump(all_results, f, indent=4)
    
    # Create comparison plot
    create_comparison_plot(all_results, results_dir)
    
    # Print summary
    print_summary(all_results)

def get_predictions_for_plot(model, test_loader, scaler, target_columns, 
                           all_columns, device):
    """Extract predictions and actuals for plotting"""
    model.eval()
    predictions = []
    actuals = []
    
def get_predictions_for_plot(model, test_loader, scaler, target_columns,
                           all_columns, device):
    """Extract predictions and actuals for plotting"""
    model.eval()
    predictions = []
    actuals = []
   
    with torch.no_grad():
        for X_batch, exogenous_batch, y_batch in test_loader:  # Unpack 3 values
            X_batch = X_batch.to(device)
            if exogenous_batch.numel() > 0:
                exogenous_batch = exogenous_batch.to(device)
            else:
                exogenous_batch = None
            
            y_pred = model(X_batch, exogenous_batch, device)  # Pass exogenous data and device
           
            # Convert to numpy
            y_pred_np = y_pred.cpu().numpy()
            y_batch_np = y_batch.cpu().numpy()
           
            # Inverse transform
            y_pred_original = inverse_transform_targets(
                y_pred_np, scaler, target_columns, all_columns
            )
            y_batch_original = inverse_transform_targets(
                y_batch_np, scaler, target_columns, all_columns
            )
           
            predictions.extend(y_pred_original.flatten())
            actuals.extend(y_batch_original.flatten())
   
    return np.array(predictions), np.array(actuals)
def create_comparison_plot(all_results, results_dir):
    """Create a comparison plot of all intervals"""
    intervals = sorted(all_results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('LSTM Performance Across Different Time Intervals', fontsize=16)
    
    metrics = ['rmse', 'mae', 'r2', 'mse']
    titles = ['RMSE', 'MAE', 'R² Score', 'MSE']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        values = [all_results[interval]['metrics']['original_scale'][metric] 
                 for interval in intervals]
        
        bars = ax.bar([f"{i} min" for i in intervals], values)
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.set_xlabel('Interval')
        
        # Color code based on performance (lower is better except R²)
        if metric == 'r2':
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        bars[best_idx].set_color('green')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "interval_comparison.png"), dpi=300)
    plt.close()

def print_summary(all_results):
    """Print a summary table of results"""
    print("\n" + "="*80)
    print("SUMMARY OF ALL INTERVALS")
    print("="*80)
    print(f"{'Interval':<12} {'Seq Len':<10} {'Context':<12} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
    print("-"*80)
    
    for interval in sorted(all_results.keys()):
        result = all_results[interval]
        print(f"{str(interval) + ' min':<12} "
              f"{result['sequence_length']:<10} "
              f"{str(result['context_hours']) + ' hours':<12} "
              f"{result['metrics']['original_scale']['rmse']:<12.2f} "
              f"{result['metrics']['original_scale']['mae']:<12.2f} "
              f"{result['metrics']['original_scale']['r2']:<12.4f}")
    
    # Find best interval for each metric
    print("\n" + "-"*80)
    print("BEST PERFORMING INTERVALS:")
    
    best_rmse = min(all_results.keys(), 
                   key=lambda x: all_results[x]['metrics']['original_scale']['rmse'])
    best_mae = min(all_results.keys(), 
                  key=lambda x: all_results[x]['metrics']['original_scale']['mae'])
    best_r2 = max(all_results.keys(), 
                 key=lambda x: all_results[x]['metrics']['original_scale']['r2'])
    
    print(f"Best RMSE: {best_rmse} min interval")
    print(f"Best MAE:  {best_mae} min interval")
    print(f"Best R²:   {best_r2} min interval")

if __name__ == "__main__":
    main()