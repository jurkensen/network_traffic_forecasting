import torch
import torch.nn as nn
from calflops import calculate_flops
from models.RNN import RNN 
from models.LSTM import LSTM
from models.Nasseri_LSTM import LSTM as LSTM_PAPER 
from models.timesFM import *

SIMPLE_LSTM = LSTM(
    input_dim=14, 
    lstm_hidden_size1=128,
    lstm_hidden_size2=64,
    num_lstm_layers=1,
    lstm_dropout=0.3,
    num_outputs=1,
    init_weights=True,
)
SIMPLE_RNN = RNN(
    input_dim= 14,  # Only main features, not exogenous
    rnn_hidden_size=128,
    num_rnn_layers=2,
    rnn_dropout=0.3,
    layer_units=[128, 64],
    num_outputs=1,
    init_weights=True,
)

PAPER_LSTM = LSTM_PAPER(
    input_dim=14, 
)

TIMES_FM = setup_timesfm()


def calculate_model_flops(model, input_size):
    flops, macs, params = calculate_flops(model=model, 
                                      input_shape=input_size,
                                      print_results=False)
    return flops, macs, params

def main():
    models = {
        "SIMPLE_LSTM": SIMPLE_LSTM,
        "SIMPLE_RNN": SIMPLE_RNN,
        "PAPER_LSTM": PAPER_LSTM
    }
    input_size = ((32, 12, 14))
    for name, model in models.items():
        flops, macs, params = calculate_model_flops(model, input_size)
        print(f"{name}:")
        print(f"  FLOPs: {flops}")
        print(f"  MACs: {macs}")
        print(f"  Params: {params}\n")
    return None


if __name__ == '__main__':
    main()