from utils.config import *
from utils.preprocessing import initial_data_stats, load_data, handle_missing_values
from models.ARIMA import ARIMAPredictor
from models.SARIMAX import SARIMAPredictor
import matplotlib.pyplot as plt
import pandas as pd 

def main():
    train_poblesec, test_poblesec = load_data(POBLESEC_ST)
    train_poblesec, test_poblesec = aggregate_data(train_poblesec), aggregate_data(test_poblesec)
    print(train_poblesec.head)


if __name__ == '__main__':
    main()