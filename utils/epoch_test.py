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
from models.LSTM import LSTM, get_predictions_for_plot

def main():
    torch.manual_seed(42)
    
    # Create results directory
    results_dir = f"epoch_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data once
    df_original = load_all_data(POBLESEC_ST)
    df_original = handle_missing_values(df_original)
    df_original = handle_outliers(df_original, 5, 95)
        
    # Aggregate data 
    interval_minutes = 5
    df_agg = aggregate_data(df_original, interval_minutes)
    df_agg = add_hour_features(df_agg, interval_minutes)    
    
    exogenous_columns = ['hour', 'hour_sin', 'hour_cos']
    df_agg, scaler = normalize_df_minmax(df_agg, feature_range=(0, 1))

    # Split data
    df_train, df_val, df_test = split_data(df_agg, 0.6, 0.2, 0.2)
    original_columns = list(df_agg.columns)

    # Dictionary to store all results
    all_results = {}
    
    epochs = [25, 50, 100, 150, 200]

    for e in epochs:
        print(f"\n{'='*60}")
        print(f"TESTING NUMBER OF EPOCHS: {e}")
        print(f"{'='*60}")
        
        # Create interval-specific directory
        interval_dir = os.path.join(results_dir, f"epoch_{e}")
        os.makedirs(interval_dir, exist_ok=True)
        
        # Create dataloaders
        exog_train = df_train[exogenous_columns]
        exog_val = df_val[exogenous_columns]
        exog_test = df_test[exogenous_columns]
        
        seq_length = int(720 / interval_minutes) # 12 hours of data at 5 min intervals

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
        model = train(model, train_loader, val_loader, epochs=e, lr=0.0001, 
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
        
        all_results[e] = results
        
        # Save individual results
        with open(os.path.join(interval_dir, "results.json"), 'w') as f:
            json.dump(results, f, indent=4)
        
        # Print results
        print(f"\n{'='*50}")
        print(f"RESULTS FOR {e} epochs")
        print(f"{'='*50}")
        print(f"Sequence Length: {seq_length} (context: {seq_length * interval_minutes / 60:.1f} hours)")
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