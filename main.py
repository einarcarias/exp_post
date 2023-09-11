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
    
    def plot_avg(self, force):
        dict = {"moment": 3, "lift": 1, "drag": 2}
        column = dict[force]
        window = 100
        smooth = self.df.iloc[:, column].rolling(window=window, center=True).mean()
        smooth.fillna(0, inplace=True)
        self.df[f"smooth {dict[force]}"] = smooth
        min_time, max_time = 0.0460, 0.0645
        relevant_data = (self.df.iloc[:, 0] >= min_time) & (self.df.iloc[:, 0] <= max_time)
        indices = self.df[relevant_data].index

        # Plotting original and smoothed data
        plt.figure(figsize=(10, 6))
        plt.plot(self.df.iloc[:, 0], self.df.iloc[:, column], label="Original Data", color="blue")
        plt.plot(self.df.iloc[:, 0], smooth, label="Moving Average", color="red")
        plt.axvline(x=min_time, color="black", linestyle="--")
        plt.axvline(x=max_time, color="black", linestyle="--")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title("Centered Moving Average Smoothing")

        plt.legend()
        plt.show()
    
    def get_avg(self, force):

        dict = {"moment": 3, "lift": 1, "drag": 2}
        column = dict[force]
        window = 100
        smooth = self.df.iloc[:, column].rolling(window=window, center=True).mean()
        smooth.fillna(0, inplace=True)
        self.df[f"smooth {dict[force]}"] = smooth
        min_time, max_time = 0.0460, 0.0645
        relevant_data = (self.df.iloc[:, 0] >= min_time) & (self.df.iloc[:, 0] <= max_time)
        indices = self.df[relevant_data].index
        std = self.df.loc[indices, [f"smooth {dict[force]}"]].std()
        avg_force = self.df.loc[indices, [f"smooth {dict[force]}"]].mean()
        return avg_force, std


    def cut_data(self, start_time, end_time):
        # reset index
        time = self.get_time_raw()
        start_index = time[time >= start_time].index[0]
        end_index = time[time <= end_time].index[-1]
        self.df = self.df.iloc[start_index : end_index + 1, :]
        self.df.reset_index(drop=True, inplace=True)

    # post proccessing methods starts here

    def find_fft(self, force):
        self.cut_data(0.02, 0.11)
        force_data = self.dict_search(force)
        N = len(force_data)
        T = self.get_time_raw()[1] - self.get_time_raw()[0]
        yf = fft(force_data.to_numpy())
        xf = fftfreq(N, T)

        # Filter out the frequencies you don't want
        bot, top = (150, 200)
        # Create a mask for the frequencies you want to zero out
        mask = (xf > bot) & (xf < top)  # You can adjust the range here
        mask = mask | (
            (xf < -top) & (xf > -top)
        )  # Also zero out the negative frequencies corresponding to this range
        yf[mask] = 0
        y_filtered = ifft(yf).real

        # using welch method
        f, Pxx_den = welch(force_data, fs=1 / T, nperseg=256)

        # top bot
        bot_butter, top_butter = (185, 195)
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




def force_coef_std_calc(dict, force, force_std):
    rho, rho_std = dict["rho"]
    velocity, velocity_std = dict["velocity"]
    diameter, diameter_std = dict["diameter"]
    force_coef = force / (0.5 * rho * velocity ** 2 * ((np.pi*diameter/4) ** 2)) 
    force_coef_std = force_coef * np.sqrt(
        (force_std / force) ** 2
        + (rho_std / rho) ** 2
        + (4* velocity_std / velocity) ** 2
        +  (4* diameter_std / diameter) ** 2
    )
    return force_coef, force_coef_std


def force_repeat_avg_calc(force_list, force_std_list):
    force_avg = np.mean(force_list)
    force_std_avg = np.sqrt(np.sum(np.array(force_std_list) ** 2)) / np.sum(
        np.array(force_list)
    ) * force_avg
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
        5: {0: "TR010.csv", 10: ["TR011.csv", "TR021.csv", "TR020.csv"]},
    }
    flap_config_data = {
        0: {5: "TR013.csv", 17.5: "TR016.csv"},
        5: {5: "TR014.csv", 17.5: ["TR015.csv", "TR017.csv", "TR019.csv"]},
    }

    rho = 0.0371
    rho_std = (7.1/100)*rho  # in percent
    velocity = 1553
    velocity_std = 1.6/100 * velocity  # in percent
    diameter = 0.06
    diameter_std = 0  # in percent. Need to ask Simon this
    # put all conditions in a dictionary
    condition_dict = {
        "rho": (rho, rho_std),
        "velocity": (velocity, velocity_std),
        "diameter": (diameter, diameter_std),
    }
    # using moving average to smooth data
    test = ExpPostProcess("TR008.csv")
    test.plot_avg("lift")

    # processing repeats
    # bicone
    bicone = {}
    bicone_repeats = {}

    for key in [0,6]:
        if type(bicone_data[key]) == list:
            for key_force in ["drag","lift","moment"]:
                force_list = []
                force_std_list = []
                for file in bicone_data[key]:
                    test = ExpPostProcess(file)
                    force, force_std = test.get_avg(key_force)
                    force_list.append(force)
                    force_std_list.append(force_std)
                force_avg, force_std_avg = force_repeat_avg_calc(force_list, force_std_list)
                force_coef, force_coef_std = force_coef_std_calc(
                    condition_dict, force_avg, force_std_avg
                )
                bicone_repeats[key_force] = (force_coef, force_coef_std)
            bicone[key] = bicone_repeats

    
    # fin config
    fin_config_repeats = {}
    

    for key_force in ["drag", "lift", "moment"]:
        force_list = []
        force_std_list = []
        for file in fin_config_data[5][10]:
            test = ExpPostProcess(file)
            force, force_std = test.get_avg(key_force)
            force_list.append(force)
            force_std_list.append(force_std)
        force_avg, force_std_avg = force_repeat_avg_calc(force_list, force_std_list)
        force_coef, force_coef_std = force_coef_std_calc(
            condition_dict, force_avg, force_std_avg
        )
        fin_config_repeats[key_force] = (force_coef, force_coef_std)
    print(f"fin config: {fin_config_repeats}")
    # flap config
    flap_config_repeats = {}
    for key_force in ["drag", "lift", "moment"]:
        force_list = []
        force_std_list = []
        for file in flap_config_data[5][17.5]:
            test = ExpPostProcess(file)
            force, force_std = test.get_avg(key_force)
            force_list.append(force)
            force_std_list.append(force_std)
        force_avg, force_std_avg = force_repeat_avg_calc(force_list, force_std_list)
        force_coef, force_coef_std = force_coef_std_calc(
            condition_dict, force_avg, force_std_avg
        )
        flap_config_repeats[key_force] = (force_coef, force_coef_std)

    print(f"flap config: {flap_config_repeats}")




