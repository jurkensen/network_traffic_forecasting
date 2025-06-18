import torch
from models.LSTM import LSTM, get_predictions_for_plot
from utils.preprocessing import *
from utils.config import *
from models.Dataloader import *
from models.train_utils import train, test_with_original_scale, inverse_transform_targets
import os 
import json

def main():
    model = LSTM(
        input_dim=14,  
        lstm_hidden_size1=128,
        lstm_hidden_size2=64,
        num_lstm_layers=1,
        lstm_dropout=0.3,
        num_outputs=1,
        init_weights=True,
        exogenous_dim=3
    )
    state_dict = torch.load('experiments/lstm_seq_length/12H/interval_5min/model_interval_5.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    lc_df = load_all_data(POBLESEC_ST)
    lc_df = handle_missing_values(lc_df)
    lc_df = handle_outliers(lc_df, 5, 95)
    
    interval_minutes = 5
    lc_agg = aggregate_data(lc_df, interval_minutes)
    lc_agg = add_hour_features(lc_agg, interval_minutes)
    exogenous_columns = ['hour', 'hour_sin', 'hour_cos']


    lc_agg, scaler = normalize_df_minmax(lc_agg, feature_range=(0, 1))
    original_columns = list(lc_agg.columns)
    lc_test_size = 1592
    lc_agg_size = len(lc_agg)
    lc_test_ratio = float(lc_test_size / lc_agg_size)
    print(lc_test_ratio)
    lc_train, lc_val, lc_test = split_data(lc_agg, 0.0, 1 - lc_test_ratio, lc_test_ratio)
    exog_test = lc_test[exogenous_columns]

    test_loader = create_dataloader(
        dataframe=lc_test,
        target_columns=['down'],
        batch_size=32,
        seq_length=144,
        exogenous_data=exog_test
    )
    test_metrics = test_with_original_scale(
        model, test_loader, nn.MSELoss(),
        device="cuda" if torch.cuda.is_available() else "cpu",
        scaler=scaler, target_columns=['down'],
        all_columns=original_columns
    )
     
    predictions, actuals = get_predictions_for_plot(
        model, test_loader, scaler, target_columns=['down'],
        all_columns=original_columns,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Create and save prediction plot
    plt.figure(figsize=(15, 6))
    plt.plot(actuals, label='Actual', color='blue', alpha=0.7)
    plt.plot(predictions, label='Predicted', color='red', alpha=0.7)
    plt.title(f'LSTM predictions for PobleSec station of a model trained on PobleSec station')
    plt.xlabel('Time Steps')
    plt.ylabel('Traffic (bytes)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("predictions.png", dpi=300)
    plt.close()

    results = {
        'interval_minutes': interval_minutes,
        'sequence_length': 144,
        'context_hours': 144 * interval_minutes / 60,
        'model_params': sum(p.numel() for p in model.parameters()),
        'metrics': test_metrics
    }
    with open(f"results_{POBLESEC_ST.name}.json", 'w') as f:
            json.dump(results, f, indent=4)

    print("\nOriginal Scale Metrics:")
    print(f"  RMSE: {test_metrics['rmse']:.2f}")
    print(f"  MAE:  {test_metrics['mae']:.2f}")
    print(f" MAPE:  {test_metrics['mape']:.2f}")
    print(f"  R²:   {test_metrics['r2']:.4f}")
    print(f"  MSE:  {test_metrics['mse']:.2f}")

    return None

if __name__ == "__main__":
    main()