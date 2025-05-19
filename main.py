from utils.config import *
from utils.preprocessing import initial_data_stats, load_data, handle_missing_values, aggregate_data
from models.ARIMA import ARIMAPredictor
from models.SARIMAX import SARIMAPredictor
import matplotlib.pyplot as plt
import pandas as pd 

def main():
    # aggregate the data to reduce noise
    train_poblesec, test_poblesec = load_data(POBLESEC_ST)
    train_poblesec, test_poblesec = aggregate_data(train_poblesec), aggregate_data(test_poblesec)
    
    arima = ARIMAPredictor()
    evaluation = arima.predict_network_traffic(train_poblesec, test_poblesec, 'down')
    
    print(evaluation)
    

    # sarima = SARIMAPredictor()
    # evaluation_sarima = sarima.predict_network_traffic(train_poblesec, test_poblesec, 'down')
    # print(evaluation_sarima)


if __name__ == '__main__':
    main()