import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.config import *
from utils.preprocessing import *
import itertools
import warnings


class SARIMAPredictor():
    
    def __init__(self):
        return

    def train_sarima_model(self, train_values, order=(1,1,1), seasonal_order=(1,1,1,24), target_column='down'):
        """
        Train a SARIMA model on the given time series
        """
        model = SARIMAX(train_values['down'], 
                        exog=train_values[['hour', 'hour_sin', 'hour_cos']],
                        order=order, 
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=True)
        
        model_fit = model.fit(disp = 0)
            
        return model_fit

    def evaluate_forecast(self, actual, predicted):
        """
        Evaluate the forecast using multiple metrics
        
        Args:
            actual (pd.Series): Actual values
            predicted (pd.Series): Predicted values
            
        Returns:
            dict: Evaluation metrics
        """
        # Ensure indexes match for comparison
        if isinstance(predicted, pd.Series):
            predicted = predicted.reindex_like(actual)
        
        # Calculate metrics
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predicted)
        
        # Calculate MAPE safely (handling zeros)
        mask = actual != 0
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }
    
    def find_best_sarima_parameters(self, train_series, max_p=2, max_d=1, max_q=2, 
                                   max_P=1, max_D=1, max_Q=1, s=24, target_column='down'):
        """
        Find the best SARIMA parameters through grid search
        
        Args:
            train_series (pd.Series): Input time series
            max_p, max_d, max_q: Maximum values for non-seasonal parameters
            max_P, max_D, max_Q: Maximum values for seasonal parameters
            s: Seasonal period (24 for hourly data)
            
        Returns:
            tuple: Best parameters (order, seasonal_order)
        """
        print("Starting grid search for optimal SARIMA parameters...")
        print("Note: This can take some time for hourly data")
        
        # Define parameter ranges
        p = range(0, max_p + 1)
        d = range(0, max_d + 1)
        q = range(0, max_q + 1)
        P = range(0, max_P + 1)
        D = range(0, max_D + 1)
        Q = range(0, max_Q + 1)
    
        # Create parameter combinations - start with simpler models
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = list(itertools.product(P, D, Q, [s]))

        best_aic = float("inf")
        best_order = None
        best_seasonal_order = None
        combinations_tried = 0

        for param in pdq:
            for seasonal_param in seasonal_pdq:     
                combinations_tried += 1
                
                try:
                    model = SARIMAX(train_series[target_column],
                                    exog=train_series[['hour', 'hour_sin', 'hour_cos']],
                                    order=param,
                                    seasonal_order=seasonal_param,
                                    enforce_stationarity=False,
                                    enforce_invertibility=True)
                     # Capture all types of warnings
                    with warnings.catch_warnings(record=True) as w:
                        model_sarima_fit = model.fit(disp=0)
                        has_warning = any(
                                issubclass(warning.category, (ConvergenceWarning, UserWarning)) for warning in w
                        )
                    
                        # Only consider models without invertibility warnings for best model selection
                        if not has_warning and model_sarima_fit.aic < best_aic:
                            best_aic = model_sarima_fit.aic
                            best_order = param
                            best_seasonal_order = seasonal_param
                            
                        print(f"{combinations_tried}: SARIMA{param}x{seasonal_param} - AIC: {model_sarima_fit.aic}{' (warning)' if has_warning else ''}")
                                                
                except Exception as e:
                    print(f"SARIMA{param}x{seasonal_param} failed with error: {e}")
                    continue
        
        print(f"Best SARIMA model: SARIMA{best_order}x{best_seasonal_order} with AIC: {best_aic}")
        return best_order, best_seasonal_order

    def predict_network_traffic(self, df_train, df_test, target_column, interval_minutes=60, order=None, seasonal_order=None):
        """
        Predict network traffic for a specific target column using SARIMA
        """     
        seasonal_period = int(1440 / interval_minutes) # For hourly data, season = 24 hours
        if order is None or seasonal_order is None:
            print("No SARIMA parameters provided, finding optimal parameters...")
            order, seasonal_order = self.find_best_sarima_parameters(
                df_train, 
                max_p=3, max_d=1, max_q=3,  # Limit non-seasonal params
                max_P=3, max_D=1, max_Q=3,  # Limit seasonal params
                s=seasonal_period,
                target_column=target_column
            )
        
        print(f"Training SARIMA model with order {order} and seasonal_order {seasonal_order}...")   
        sarima_model = self.train_sarima_model(df_train, order, seasonal_order, target_column=target_column)
        
        print(f"Generating forecast for {df_test} steps...")
        raw_prediction = sarima_model.forecast(steps=len(df_test), exog=df_test[['hour', 'hour_sin', 'hour_cos']])
        
        return raw_prediction, order, seasonal_order


# def main():
#     sarima = SARIMAPredictor()
#     df_train, df_val, df_test = load_data(POBLESEC_ST)
#     df_train, df_val, df_test  = handle_missing_values(df_train), handle_missing_values(df_val), handle_missing_values(df_test)
#     df_train, df_val, df_test = aggregate_data(df_train), aggregate_data(df_val), aggregate_data(df_test)
#     df_train, df_val, df_test = sarima.add_hour_features(df_train), sarima.add_hour_features(df_val), sarima.add_hour_features(df_test)

#     df_all = pd.concat([df_train, df_val, df_test])
#     df_all.index.freq = pd.infer_freq(df_all.index)
#     print("Aggregated dataframe shape:", df_all.shape)
#     # df_all, scaler = normalize_column(df_all, 'down')
    
#     # Define window size (1 day = 24 hours)
#     window_size = 24
#     # Number of sliding steps (for test purposes, run for 3 hours)
#     num_steps = 3

#     for i in range(num_steps):
#         start_idx = i
#         end_idx = i + window_size
#         test_idx = end_idx  # Predict the next hour

#         train_window = df_all.iloc[start_idx:end_idx]
#         test_window = df_all.iloc[test_idx:test_idx+1]

#         print(f"\nSliding window {i+1}: Training on {train_window.index[0]} to {train_window.index[-1]}, predicting {test_window.index[0]}")

#         # Fit SARIMA and predict
#         result = sarima.predict_network_traffic(
#             train_window,
#             test_window,
#             target_column='down',
#             use_hour_stats=True
#         )

#         print(f"Prediction: {result['predictions'].values}, Actual: {result['actual'].values}")


# main()    
