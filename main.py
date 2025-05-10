from utils.config import *
from utils.preprocessing import initial_data_stats, load_data, handle_missing_values
from models.ARIMA import ARIMAPredictor
import matplotlib.pyplot as plt

def main():

    arima = ARIMAPredictor()
    train_poblesec, test_poblesec = load_data(POBLESEC_ST)
    
    ## HANDLE MISSING DATA
    
    # print("\n===== PREDICTING DOWNLINK TRAFFIC =====")
    # down_results = arima.predict_network_traffic(train_poblesec, test_poblesec, target_column='down')
    
    # # Predict uplink traffic
    # print("\n===== PREDICTING UPLINK TRAFFIC =====")
    # up_results = arima.predict_network_traffic(train_poblesec, test_poblesec, target_column='up')
    
    # # Print summary
    # print("\n===== SUMMARY OF RESULTS =====")
    # print(f"Downlink Traffic RMSE: {down_results['evaluation']['RMSE']:.4f}")
    # print(f"Uplink Traffic RMSE: {up_results['evaluation']['RMSE']:.4f}")
    


if __name__ == '__main__':
    main()