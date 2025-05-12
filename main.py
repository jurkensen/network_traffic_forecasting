from utils.config import *
from utils.preprocessing import initial_data_stats, load_data, handle_missing_values
from models.ARIMA import ARIMAPredictor
import matplotlib.pyplot as plt

def main():

    arima = ARIMAPredictor()
    ## ---- LOADING AND PREPROCESSING DATA ------
    print("\n===== LOADING AND PREPROCESSING DATA =====")
    train_poblesec, test_poblesec = load_data(POBLESEC_ST)   
    plt.figure(figsize=(20,4))
    ax = plt.gca()
    ax.set_xlabel('Time')
    ax.set_ylabel('DownLink bitrate')
    plt.plot(test_poblesec.index, test_poblesec['down']);  
    # plt.show()
    

    stats = arima.check_stationarity(train_poblesec['down'])
    # print(stats)

    # print("\n===== PREDICTING DOWNLINK TRAFFIC =====")
    # down_results = arima.predict_network_traffic(train_poblesec, test_poblesec, target_column='down')
    print("\n===== PREDICTING UPLINK TRAFFIC =====")
    up_results = arima.predict_network_traffic(train_poblesec, test_poblesec, target_column='up')
    
    # Print summary
    print("\n===== SUMMARY OF RESULTS =====")
    print(f"Downlink Traffic RMSE: {down_results['RMSE']:.4f}")
    print(f"Uplink Traffic RMSE: {up_results['RMSE']:.4f}")
    


if __name__ == '__main__':
    main()