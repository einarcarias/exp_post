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
    window = 90
    smooth = data.iloc[:, column].rolling(window=window, center=True).mean()
    smooth.fillna(0, inplace=True)
    data["smooth"] = smooth
    min_max_dict = {1: (0.0495, 0.0659), 2: (0.0460, 0.0665), 3: (0.0479, 0.0663)}
    min_time, max_time = min_max_dict[column]
    relevant_data = (data.iloc[:, 0] >= min_time) & (data.iloc[:, 0] <= max_time)
    indices = data[relevant_data].index
    std = data.loc[indices, ["smooth"]].std()
    avg_force = data.loc[indices, ["smooth"]].mean()

    # Plotting original and smoothed data
    plt.figure(figsize=(10, 6))
    plt.plot(data.iloc[:, 0], data.iloc[:, column], label="Original Data", color="blue")
    plt.plot(data.iloc[:, 0], smooth, label="Moving Average", color="red")
    plt.axvline(x=min_time, color="black", linestyle="--")
    plt.axvline(x=max_time, color="black", linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Centered Moving Average Smoothing")

    plt.legend()
    plt.show()
    os.chdir("..")
    print(avg_force)
    return avg_force, std


def force_coef_std(dict, force, force_std):
    rho, rho_std = dict["rho"]
    velocity, velocity_std = dict["velocity"]
    diameter, diameter_std = dict["diameter"]
    force_coef = (force - rho * velocity**2 * diameter**2 / 8) / (
        rho * velocity**2 * diameter**2 / 8
    )
    force_coef_std = force_coef * np.sqrt(
        (force_std / force) ** 2
        + (rho_std / rho) ** 2
        + (2 * velocity_std / velocity) ** 2
        + (diameter_std / diameter) ** 2
    )
    return force_coef, force_coef_std


def force_repeat_avg(force_list, force_std_list):
    force_avg = np.mean(force_list)
    force_std_avg = np.sqrt(np.sum(np.array(force_std_list) ** 2)) / np.sum(
        np.array(force_list)
    )
    return force_avg, force_std_avg


if __name__ == "__main__":
    column = 3
    list_repeats = []
    a5d0 = read_data(column, "TR001.csv")
    rho = 0.0371
    rho_std = 7.1  # in percent
    velocity = 1553
    velocity_std = 1.6  # in percent
    diameter = 0.06
    diameter_std = 0  # in percent. Need to ask Simon this
    # put all conditions in a dictionary
    condition_dict = {
        "rho": (rho, rho_std),
        "velocity": (velocity, velocity_std),
        "diameter": (diameter, diameter_std),
    }
    # using moving average to smooth data
