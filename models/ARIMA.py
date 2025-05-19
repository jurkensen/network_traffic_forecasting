import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.config import *
from utils.preprocessing import *
import itertools
import warnings


class ARIMAPredictor():
    
    def __init__(self):
        return

    def train_arima_model(self, train_values, order=(0,0,0)):
        """
        Train an ARIMA model on the given time series
        
        Args:
            series (pd.Series): Input time series
            order (tuple): ARIMA model order (p,d,q) parameters
            
        Returns:
            ARIMA model: Trained ARIMA model
        """
        model = ARIMA(train_values, order = order)
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
        # Calculate metrics
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }
    
    def find_best_arima_parameters(self, train_series):
        p=q=range(0,6)
        d=range(0,3)
        pdq = list(itertools.product(p,d,q))
        
        lowest_aic = float("inf")
        best_parameters = (1,1,1) # default to an easy model 
        
        for param in pdq:
            try:
                model_arima = ARIMA(train_series, order=param, enforce_invertibility=True)

                # Capture all types of warnings
                with warnings.catch_warnings(record=True) as w:
                    model_arima_fit = model_arima.fit()
                    
                    has_convergence_warning = any(
                        issubclass(warning.category, ConvergenceWarning) for warning in w
                    )
                
                # Only consider models without invertibility warnings for best model selection
                if not has_convergence_warning and model_arima_fit.aic < lowest_aic:
                    lowest_aic = model_arima_fit.aic
                    best_parameters = param
                    
                print(f"{param} - AIC: {model_arima_fit.aic}{' (convergence warning)' if has_convergence_warning else ''}") 
            except Exception as e:
                print(f"ARIMA was not able to fit to the data with parameters: {param}, due to Exception:{e}" )
                continue
        
        return best_parameters
    
    def predict_network_traffic(self, df_train, df_test, target_column):
        """
        Predict network traffic for a specific target column
        
        Args:
            df_train (pd.DataFrame): data on which the predictions are made
            df_test (pd.DataFrame): test data, predictions accuracy checked against this series
            
        Returns:
            dict: Evaluation metrics
        """
        
        train_series = df_train[target_column]
        test_series = df_test[target_column]

        print("Finding the optimal ARIMA parameters..")
        # order = self.find_best_arima_parameters(train_series)
        order = (2,0,3)
        print(f"Training ARIMA model with order {order}...")    
        arima_model = self.train_arima_model(train_series, order)

        print(f"Generating forecast for {len(test_series)} steps...")
        raw_prediction = arima_model.forecast(steps=len(test_series))    
        
        prediction = pd.Series(
            data=raw_prediction,
            index=test_series.index,
            name='forecast'
        )
        
        prediction = prediction.ffill().bfill()

        evaluation = self.evaluate_forecast(test_series, prediction)
        print(f"\nEvaluation metrics for {target_column} traffic:")
        for metric, value in evaluation.items():
            print(f"  {metric}: {value:.4f}")

        # Plot the forecast vs actual
        plt.figure(figsize=(15, 7))
        plt.plot(test_series.values, label='Actual')
        plt.plot(prediction.values, label='Forecast', color='red')
        plt.title(f'ARIMA (2,0,3) Forecast vs Actual for {target_column.capitalize()} Traffic')
        plt.xlabel('Time Steps')
        plt.ylabel('Bitrate')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'arima_forecast_{target_column}.png')
        plt.close()

        return evaluation
    
    def check_stationarity(self, ts):
        stats = ['Test Statistic','p-value','Lags','Observations']
        df_test = adfuller(ts, autolag='AIC')
        df_results = pd.Series(df_test[0:4], index=stats)
        for key,value in df_test[4].items():
            df_results['Critical Value (%s)'%key] = value
        
        return df_results