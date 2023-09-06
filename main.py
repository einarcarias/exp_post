from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
import os
from statsmodels.tsa.seasonal import STL
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import welch
from scipy.signal import butter, filtfilt
import numpy as np


class ExpPostProcess:
    data_dir = Path("data")

    def __init__(self, filename):
        os.chdir(self.data_dir)
        self.df = pd.read_csv(filename, delimiter="\t")
        os.chdir("..")

    # getters for raw data
    def get_moment_raw(self):
        return self.df.iloc[:, 3]

    def get_lift_raw(self):
        return self.df.iloc[:, 1]

    def get_time_raw(self):
        return self.df.iloc[:, 0]

    def get_drag_raw(self):
        return self.df.iloc[:, 2]

    def dict_search(self, force):
        force_dict = {
            "moment": self.get_moment_raw(),
            "lift": self.get_lift_raw(),
            "drag": self.get_drag_raw(),
        }
        return force_dict[force]

    # post proccessing methods starts here

    def find_fft(self, force):
        force_data = self.dict_search(force)
        N = len(force_data)
        T = self.get_time_raw()[1] - self.get_time_raw()[0]
        yf = fft(force_data.to_numpy())
        xf = fftfreq(N, T)

        # Filter out the frequencies you don't want
        bot, top = (52, 59)
        # Create a mask for the frequencies you want to zero out
        mask = (xf > bot) & (xf < top)  # You can adjust the range here
        mask = mask | (
            (xf < -top) & (xf > -top)
        )  # Also zero out the negative frequencies corresponding to this range
        yf[mask] = 0
        y_filtered = ifft(yf).real

        # using welch method
        f, Pxx_den = welch(force_data, fs=1 / T, nperseg=1000)

        # top bot
        bot_butter, top_butter = (44, 67)
        # apply butterworth filter from welch method
        y_filtered_butter = ExpPostProcess.butter_bandpass_filter(
            force_data, bot_butter, top_butter, 1 / T, order=6
        )

        # Plot original time series
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.title("Original Time Series")
        plt.plot(self.get_time_raw(), force_data)
        plt.xlabel("Time")
        plt.ylabel(force)

        # Plot Fourier Transform results
        plt.subplot(3, 1, 2)
        plt.title("Fourier Transform")
        plt.plot(xf[: N // 2], 2.0 / N * np.abs(yf[: N // 2]))
        plt.xlabel("Frequency")
        plt.ylabel("Amplitude")

        # plot inverse Fourier Transform
        plt.subplot(3, 1, 3)
        plt.title("Inverse Fourier Transform")
        plt.plot(self.get_time_raw(), y_filtered)
        plt.xlabel("Time")
        plt.ylabel(force)

        plt.tight_layout()
        plt.show()

        # Plot original time series
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.title("Original Time Series")
        plt.plot(self.get_time_raw(), force_data)
        plt.xlabel("Time")
        plt.ylabel(force)
        # plot welch method
        plt.subplot(3, 1, 2)
        plt.title("Welch Method")
        plt.plot(f, Pxx_den)
        plt.xlabel("Frequency")
        plt.ylabel("PSD")

        # plot welch method with butterworth filter
        plt.subplot(3, 1, 3)
        plt.title("Welch Method with Butterworth Filter")
        plt.plot(self.get_time_raw(), y_filtered_butter)
        plt.xlabel("Time")
        plt.ylabel(force)
        plt.tight_layout()
        plt.show()

    @staticmethod
    # Function to design Butterworth bandpass filter
    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype="band")
        return b, a

    @staticmethod
    # Function to apply the filter
    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = ExpPostProcess.butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data)
        return y


def read_data(column, filename):
    data_dir = Path("data")
    os.chdir(data_dir)

    data = pd.read_csv(filename, delimiter="\t")
    window = 90
    smooth = data.iloc[:, column].rolling(window=window, center=True).mean()
    smooth.fillna(0, inplace=True)
    data["smooth"] = smooth
    min_max_dict = {1: (0.0460, 0.0665), 2: (0.0460, 0.0665), 3: (0.0460, 0.0665)}
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
    # data file dictionary
    # modify accordingly
    bicone_data = {
        0: ["TR001.csv", "TR007.csv", "TR008.csv"],
        2: "TR002.csv",
        4: "TR003.csv",
        6: ["TR004.csv", "TR005.csv", "TR006.csv"],
    }
    fin_config_data = {
        0: {0: "TR009.csv", 10: "TR012.csv"},
        5: {0: "TR010.csv", 10: "TR011.csv"},
    }
    flap_config_data = {
        0: {5: "TR013.csv", 17.5: "TR016.csv"},
        5: {5: "TR014.csv", 17.5: ["TR015.csv", "TR017.csv"]},
    }

    column = 2
    list_repeats = []
    a5d0 = read_data(column, "TR004.csv")
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
    test = ExpPostProcess("TR004.csv")
    test.find_fft("lift")
    # using moving average to smooth data
