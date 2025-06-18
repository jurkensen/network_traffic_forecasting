import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity
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
def profile_timesfm_forecast(tfm, context_length=2016, num_predictions=1):
    """Profile TimesFM using its forecast method"""
    
    # Create dummy input
    dummy_input = torch.randn(1, context_length)
    freq = [0]  # Dummy frequency
    
    # Warm up
    with torch.no_grad():
        _ = tfm.forecast(inputs=dummy_input, freq=freq, normalize=True)
    
    # Profile the forecast
    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_flops=True,
        with_stack=True
    ) as prof:
        with torch.no_grad():
            for _ in range(num_predictions):
                _ = tfm.forecast(inputs=dummy_input, freq=freq, normalize=True)
    
    # Analyze results
    print("=== TimesFM Profiling Results ===")
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total",
        row_limit=20
    ))
    
    # Calculate total FLOPs
    events = prof.events()
    total_flops = sum([event.flops for event in events if event.flops > 0])
    
    print(f"\nTotal FLOPs per forecast: {total_flops:,}")
    print(f"FLOPs in GFLOPs: {total_flops / 1e9:.2f}")
    
    return total_flops / num_predictions  

def main():
    # models = {
    #     "SIMPLE_LSTM": SIMPLE_LSTM,
    #     "SIMPLE_RNN": SIMPLE_RNN,
    #     "PAPER_LSTM": PAPER_LSTM
    # }
    # input_size = ((32, 12, 14))
    # for name, model in models.items():
    #     flops, macs, params = calculate_model_flops(model, input_size)
    #     print(f"{name}:")
    #     print(f"  FLOPs: {flops}")
    #     print(f"  MACs: {macs}")
    #     print(f"  Params: {params}\n")
    
    TIMES_FM = setup_timesfm()
    profile_timesfm_forecast(TIMES_FM, 2016, 5)
    
if __name__ == '__main__':
    main()