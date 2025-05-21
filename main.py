from utils.config import *
from utils.preprocessing import initial_data_stats, load_data, handle_missing_values, aggregate_data, preprocess_outliers
from models.ARIMA import ARIMAPredictor
from models.SARIMAX import SARIMAPredictor
from models.RNN import *
import matplotlib.pyplot as plt
import pandas as pd 
import torch

def main():
    # aggregate the data to reduce noise
    train_poblesec, test_poblesec = load_data(POBLESEC_ST)
    train_poblesec, test_poblesec = aggregate_data(train_poblesec), aggregate_data(test_poblesec)
    train_poblesec, test_poblesec = preprocess_outliers(train_poblesec), preprocess_outliers(test_poblesec)
    normalized_train_data, scaler = normalize_df_minmax(train_poblesec)

    normalized_test_data = pd.DataFrame(
        scaler.transform(test_poblesec),
        index=test_poblesec.index,
        columns=test_poblesec.columns
    )

    test_dataset = NetworkDataset(normalized_test_data, 'down')

    test_loader = create_dataloader(
        test_dataset, 
        batch_size=32,  
        seq_length=12,
        forecast_horizon=1
    )

    input_size = len(train_poblesec.columns) 
    output_size = 1 
    model = RNN_MLP(input_size, output_size)    
    model.load_state_dict(torch.load('results/rnn_mlp_model.pth', map_location=torch.device('cpu')))
    model.eval()

    evaluation = evaluate_model(model, train_poblesec, test_loader, scaler)
    print(evaluation)
    plot_predictions(evaluation)

if __name__ == '__main__':
    main()