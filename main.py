from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
import os
from statsmodels.tsa.seasonal import STL
from scipy.fft import fft, fftfreq
import numpy as np

if __name__ == "__main__":
    data_dir = Path("data")
    os.chdir(data_dir)
    data = pd.read_csv("TR005.csv", delimiter="\t")
    column = 2
    # using moving average to smooth data
    window = 100
    shift = window // 2
    test = data.iloc[:, column].rolling(window=window).mean().shift(-shift)
    test.fillna(0, inplace=True)
    min_time = 0.0456
    max_time = 0.065
    relevant_data = (data.iloc[:, 0] >= min_time) & (data.iloc[:, 0] <= max_time)
    indices = data[relevant_data].index
    avg_force = data.iloc[indices, column].mean()
    # using stl decomposition
    stl = STL(data.iloc[:, column], period=80, seasonal=15, robust=True)
    res = stl.fit()
    trend = res.trend

    # Plotting original and smoothed data
    plt.figure(figsize=(10, 6))
    plt.plot(data.iloc[:, 0], data.iloc[:, column], label="Original Data", color="blue")
    plt.plot(data.iloc[:, 0], test, label="Moving Average", color="red")
    plt.plot(data.iloc[:, 0], trend, label="STL", color="green")
    plt.axvline(x=min_time, color="black", linestyle="--")
    plt.axvline(x=max_time, color="black", linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Centered Moving Average Smoothing")

    plt.legend()
    plt.show()
    print(f"Average force: {avg_force}")
