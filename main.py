from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
import os
from statsmodels.tsa.seasonal import STL
from scipy.fft import fft, fftfreq
import numpy as np


def read_data(column, filename):
    data_dir = Path("data")
    os.chdir(data_dir)

    data = pd.read_csv(filename, delimiter="\t")
    window = 100
    shift = window // 2
    test = data.iloc[:, column].rolling(window=window).mean().shift(-shift)
    test.fillna(0, inplace=True)
    min_time = 0.0450
    max_time = 0.0620
    relevant_data = (data.iloc[:, 0] >= min_time) & (data.iloc[:, 0] <= max_time)
    indices = data[relevant_data].index
    avg_force = data.iloc[indices, column].mean()

    # Plotting original and smoothed data
    plt.figure(figsize=(10, 6))
    plt.plot(data.iloc[:, 0], data.iloc[:, column], label="Original Data", color="blue")
    plt.plot(data.iloc[:, 0], test, label="Moving Average", color="red")
    plt.axvline(x=min_time, color="black", linestyle="--")
    plt.axvline(x=max_time, color="black", linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Centered Moving Average Smoothing")

    plt.legend()
    plt.show()
    os.chdir("..")
    print(avg_force)
    return avg_force


if __name__ == "__main__":
    column = 2
    a5d0 = read_data(column, "TR006.csv")
    # using moving average to smooth data
