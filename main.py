from utils.config import *
from utils.preprocessing import initial_data_stats, load_data, handle_missing_values
from models.ARIMA import ARIMAPredictor
from models.SARIMAX import SARIMAPredictor
import matplotlib.pyplot as plt
import pandas as pd 

def main():
    sarima = SARIMAPredictor()
    
    print("\n===== LOADING DATA =====")
    train_poblesec, test_poblesec = load_data(POBLESEC_ST)

    print("\n===== PREDICTING DOWNLINK TRAFFIC WITH SEASONALITY =====")
    down_results = sarima.predict_network_traffic(train_poblesec, test_poblesec, target_column='down')
    
    # Optional: Also predict uplink traffic
    # print("\n===== PREDICTING UPLINK TRAFFIC WITH SEASONALITY =====")
    # up_results = sarima.predict_network_traffic(train_poblesec, test_poblesec, target_column='up')
    
    # Print summary
    print("\n===== SUMMARY OF RESULTS =====")
    print(f"Downlink Traffic RMSE: {down_results['evaluation']['RMSE']:.4f}")
    # print(f"Uplink Traffic RMSE: {up_results['evaluation']['RMSE']:.4f}")
    
    # Compare with the previous ARIMA results
    print("\n===== COMPARING WITH ORIGINAL ARIMA RESULTS =====")
    print("Previous ARIMA RMSE: 139823620.3378")
    print(f"SARIMA RMSE: {down_results['evaluation']['RMSE']:.4f}")
    
    improvement = (139823620.3378 - down_results['evaluation']['RMSE']) / 139823620.3378 * 100
    print(f"Improvement: {improvement:.2f}%")

    # arima = ARIMAPredictor()
    # ## ---- LOADING AND PREPROCESSING DATA ------
    # print("\n===== LOADING AND PREPROCESSING DATA =====")
    # train_poblesec, test_poblesec = load_data(POBLESEC_ST)   
    # plt.figure(figsize=(20,4))
    # plt.plot(train_poblesec, label='Train Set')
    # plt.plot(test_poblesec, label='Test Set', color='orange')

    # plt.plot(train_poblesec.index, train_poblesec['down']);  
    # plt.show()

    # print("\n===== PREDICTING DOWNLINK TRAFFIC =====")
    # down_results = arima.predict_network_traffic(train_poblesec, test_poblesec, target_column='down')
    # print("\n===== PREDICTING UPLINK TRAFFIC =====")
    # up_results = arima.predict_network_traffic(train_poblesec, test_poblesec, target_column='up')
    
    # # Print summary
    # print("\n===== SUMMARY OF RESULTS =====")
    # print(f"Downlink Traffic RMSE: {down_results['RMSE']:.4f}")
    # print(f"Uplink Traffic RMSE: {up_results['RMSE']:.4f}")
    


if __name__ == '__main__':
    main()