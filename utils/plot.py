import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot(series, col_name, title="Time Series Data"):
    plt.figure(figsize=(20,4))
    plt.plot(series[col_name], label='Data', color='blue')
    # plt.plot(val_series[col_name], label='Validation Set', color = 'green')
    # plt.plot(test_series[col_name], label='Test Set', color='red')
    plt.xlabel("Time")
    plt.ylabel(f"{col_name}")
    plt.legend()
    plt.show()
    plt.savefig(f"{title}_{col_name}.png")

def plot_weekday_means(df, col_name='down', title='Mean Downlink by Day of Week'):
    df['weekday'] = df.index.day_name()
    weekday_means = df.groupby('weekday')[col_name].mean()
    # Ensure correct weekday order
    weekday_means = weekday_means.reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ])
    plt.figure(figsize=(10,5))
    weekday_means.plot(kind='bar', color='skyblue')
    plt.title(title)
    plt.ylabel('Mean Downlink')
    plt.xlabel('Day of Week')
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{title.replace(' ', '_')}.png")

def plot_forecast(test_set, prediction, order, seasonal_order, target_column, interval_minutes=1, file_name='sarima_forecast'):
    # Plot the forecast vs actual
    plt.figure(figsize=(15, 7))
    plt.plot(test_set, label='Actual')
    plt.plot(prediction, label='Forecast', color='red')
    plt.title(f'SARIMA {order}, {seasonal_order} Forecast vs Actual for {target_column.capitalize()} Traffic for Interval {interval_minutes} minutes')
    plt.xlabel('Time Steps')
    plt.ylabel('Bitrate')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{file_name}.png')
    plt.close()