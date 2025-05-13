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

    def train_sarima_model(self, train_values, order=(1,1,1), seasonal_order=(1,1,1,24)):
        """
        Train a SARIMA model on the given time series
        
        Args:
            train_values (pd.Series): Input time series
            order (tuple): ARIMA model order (p,d,q) parameters
            seasonal_order (tuple): Seasonal components (P,D,Q,s) where s is the season length
            
        Returns:
            SARIMA model: Trained SARIMA model
        """
        model = SARIMAX(train_values, 
                        order=order, 
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=True)
        
        model_fit = model.fit()
            
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
                                   max_P=1, max_D=1, max_Q=1, s=24):
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
        
        # Sort combinations by complexity (sum of parameters)
        pdq = sorted(pdq, key=lambda x: sum(x))
        seasonal_pdq = sorted(seasonal_pdq, key=lambda x: sum(x[:3]))
        
        best_aic = float("inf")
        best_order = None
        best_seasonal_order = None
        
        # Limit the combinations to avoid excessive computation
        max_combinations = 3000  # Adjust based on available computation time
        combinations_tried = 0
        
        # Get a sample of the data for faster parameter search
        sample_size = min(len(train_series), 1000)  # Use at most 1000 data points for parameter search
        train_sample = train_series.iloc[-sample_size:]
        
        for param in pdq:
            if combinations_tried >= max_combinations:
                break
                
            for seasonal_param in seasonal_pdq:
                if combinations_tried >= max_combinations:
                    break
                    
                combinations_tried += 1
                
                try:
                    model = SARIMAX(train_sample,
                                  order=param,
                                  seasonal_order=seasonal_param,
                                  enforce_stationarity=False,
                                  enforce_invertibility=True)
                    # Capture all types of warnings
                    with warnings.catch_warnings(record=True) as w:
                        model_arima_fit = model.fit(disp=0)
                        
                        # Check if we got invertibility warnings
                        has_invertibility_warning = any(
                            "Non-invertible" in str(warning.message) for warning in w
                        )
                    aic = model_arima_fit.aic
                    # Only consider models without invertibility warnings for best model selection
                    if not has_invertibility_warning and aic < best_aic:
                        best_aic = aic
                        best_order = param
                        best_seasonal_order = seasonal_param
                    
                    print(f"{combinations_tried}: SARIMA{param}x{seasonal_param} - AIC: {aic}{' (Non-invertible warning)' if has_invertibility_warning else ''}")
                                            
                        
                except Exception as e:
                    print(f"SARIMA{param}x{seasonal_param} failed with error: {e}")
                    continue
        
        print(f"Best SARIMA model: SARIMA{best_order}x{best_seasonal_order} with AIC: {best_aic}")
        return best_order, best_seasonal_order

    def add_hour_features(self, df):
        """
        Add hour-of-day features to the dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe with datetime index
            
        Returns:
            pd.DataFrame: Dataframe with added hour features
        """                
        # Add hour features
        result = df.copy()
        result['hour'] = result.index.hour
        
        # Add hour as cyclical features (sin/cos transformation)
        # This helps the model understand the circular nature of hours
        hours_in_day = 24
        result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / hours_in_day)
        result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / hours_in_day)
        
        return result
    
    def predict_network_traffic(self, df_train, df_test, target_column, use_hour_stats=True):
        """
        Predict network traffic for a specific target column using SARIMA
        
        Args:
            df_train (pd.DataFrame): data on which the predictions are made
            df_test (pd.DataFrame): test data, predictions accuracy checked against this series
            target_column (str): Column name to predict ('down' or 'up')
            use_hour_stats (bool): Whether to use hourly stats as exogenous variables
            
        Returns:
            dict: Evaluation metrics and predictions
        """     
        train_series = df_train[target_column]
        test_series = df_test[target_column]
        
        # Check seasonality in the data
        print("Checking seasonality patterns...")
        df_train_with_hour = self.add_hour_features(df_train)
        hour_stats = df_train_with_hour.groupby('hour').mean()[target_column]
        
        seasonal_period = 24  # For hourly data, season = 24 hours
        
        print("Finding the optimal SARIMA parameters...")
        order, seasonal_order = self.find_best_sarima_parameters(
            train_series, 
            max_p=5, max_d=2, max_q=5,  # Limit non-seasonal params
            max_P=5, max_D=2, max_Q=5,  # Limit seasonal params
            s=seasonal_period
        )
        
        print(f"Training SARIMA model with order {order} and seasonal_order {seasonal_order}...")   
        sarima_model = self.train_sarima_model(train_series, order, seasonal_order)
        
        print(f"Generating forecast for {len(test_series)} steps...")
        prediction = sarima_model.forecast(steps=len(test_series))
        
        # Reset indices for comparison
        prediction = pd.Series(prediction).reset_index(drop=True)
        test_actual = test_series.reset_index(drop=True)
        
        # Evaluate the forecast
        evaluation = self.evaluate_forecast(test_actual, prediction)
        print(f"\nEvaluation metrics for {target_column} traffic:")
        for metric, value in evaluation.items():
            print(f"  {metric}: {value:.4f}")
        
        # Plot the forecast vs actual
        plt.figure(figsize=(15, 7))
        plt.plot(test_actual.values, label='Actual')
        plt.plot(prediction.values, label='Forecast', color='red')
        plt.title(f'SARIMA Forecast vs Actual for {target_column.capitalize()} Traffic')
        plt.xlabel('Time Steps')
        plt.ylabel('Traffic Volume')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'sarima_forecast_{target_column}.png')
        plt.close()
        
        # Return results
        return {
            'evaluation': evaluation,
            'predictions': prediction,
            'actual': test_actual,
            'model': sarima_model,
            'hourly_pattern': hour_stats
        }


def main():
    """
    Main function to run the SARIMA prediction
    """
    