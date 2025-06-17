import timesfm
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.preprocessing import *

def setup_timesfm():
    """Initialize TimesFM model"""
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="gpu" if torch.cuda.is_available() else "cpu",
            per_core_batch_size=32,
            horizon_len=1, 
            num_layers=50,
            context_len=2016,  # Max historical context
            use_positional_embedding=False,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
        ),
    )
    return tfm

def evaluate(actual_targets, forecast, interval_minutes=5):
    rmse = np.sqrt(mean_squared_error(actual_targets, forecast))
    mae = mean_absolute_error(actual_targets, forecast)
    r2 = r2_score(actual_targets, forecast)
    
    print(f"TimesFM Results for {interval_minutes}-minute intervals:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"R²:   {r2:.4f}")
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2}


def main():
    df_original = load_all_data(POBLESEC_ST)
    df_original = load_all_data(POBLESEC_ST)
    df_original = handle_missing_values(df_original)
    df_original = handle_outliers(df_original, 5, 95)

    interval_minutes = 5
    df_agg = aggregate_data(df_original, interval_minutes)
    df_agg = add_hour_features(df_agg, interval_minutes)

    tfm = setup_timesfm()
    
    df_train, df_val, df_test = split_data(df_agg, 0.8, 0.0, 0.2)
    
    context_series = df_train['down'].values 
    target_series = df_test['down'].values
    
    context_length = min(len(df_train), 2016) # max for timesfm, 24h of context
        
    # TimesFM prediction
    frequency = [0] 
    predictions = []
    for i in range(len(target_series)): 
        current_context = np.concatenate([context_series, target_series[:i]])

        current_context = current_context[-context_length:]
        context_tensor = torch.tensor(current_context, dtype=torch.float32).unsqueeze(0) 
        try:
            mean_forecast, _  = tfm.forecast(
                inputs=context_tensor,
                freq=frequency,
                normalize=True
            )
            predicted_value = mean_forecast.item()
            predictions.append(predicted_value)
        except Exception as e:
            print(f"TimesFM failed at step {i}: {e}")
            break
    if len(target_series) != len(predictions):
        print(f"Something's wrong, {len(target_series)}:{len(predictions)}")
        return None
    evaluation = evaluate(target_series, predictions, 5)



if __name__ == "__main__":
    main()