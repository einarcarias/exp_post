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

    def __init__(self, filename, aoa):
        self.aoa = aoa
        os.chdir(self.data_dir)
        self.filename = filename
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
        self.get_avg(force)
        min_time, max_time = {
            1: (0.0495, 0.0645),
            2: (0.0460, 0.0645),
            3: (0.0460, 0.0645),
        }[column]

        # Plotting original and smoothed data
        plt.figure(figsize=(6, 3))
        plt.plot(
            self.df.iloc[:, 0],
            self.df.iloc[:, column],
            label="Original Data",
            color="blue",
        )
        plt.plot(
            self.df.iloc[:, 0],
            self.df[f"smooth {dict[force]}"],
            label="Moving Average",
            color="red",
        )
        plt.axvline(x=min_time, color="black", linestyle="--")
        plt.axvline(x=max_time, color="black", linestyle="--")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title(f"Centered Moving Average Smoothing for {self.name()}")

        plt.legend()
        plt.tight_layout()
        plt.show(block=False)

    def name(self):
        bicone_data = {
            0: ["TR001.csv", "TR007.csv", "TR008.csv"],
            2: ["TR002.csv"],
            4: ["TR003.csv"],
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

        for aoa, filenames in bicone_data.items():
            if self.filename in filenames:
                return f"AoA = {aoa}, Bicone"

        for aoa, delta_data in fin_config_data.items():
            for delta, filenames in delta_data.items():
                filenames = filenames if isinstance(filenames, list) else [filenames]
                if self.filename in filenames:
                    return f"AoA = {aoa}, Delta = {delta} Fin"

        for aoa, delta_data in flap_config_data.items():
            for delta, filenames in delta_data.items():
                filenames = filenames if isinstance(filenames, list) else [filenames]
                if self.filename in filenames:
                    return f"AoA = {aoa}, Delta = {delta} Flap"

        return "Unknown"

    def get_avg(self, force):
        dict = {"moment": 3, "lift": 1, "drag": 2}
        column = dict[force]
        window = 50
        smooth = self.df.iloc[:, column].rolling(window=window, center=True).mean()
        smooth.fillna(0, inplace=True)
        self.df[f"smooth {dict[force]}"] = smooth
        min_time, max_time = {
            1: (0.0495, 0.0645) if self.aoa != 0 else (0.05, 0.07),
            2: (0.0460, 0.0645),
            3: (0.0460, 0.0645),
        }[column]
        relevant_data = (self.df.iloc[:, 0] >= min_time) & (
            self.df.iloc[:, 0] <= max_time
        )
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
        self.cut_data(0.040, 0.07)
        force_data = self.dict_search(force)
        N = len(force_data)
        Fs = 1 / np.mean(np.diff(self.get_time_raw()))
        T = 1 / Fs
        yf = fft(force_data.to_numpy())[: N // 2] / N
        xf = fftfreq(N, T)[: N // 2]

        # Filter out the frequencies you don't want
        bot, top = (150, 200)
        # Create a mask for the frequencies you want to zero out
        mask = (xf > bot) & (xf < top)  # You can adjust the range here
        mask = mask | (
            (xf < -top) & (xf > -top)
        )  # Also zero out the negative frequencies corresponding to this range
        yf[mask] = 0

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
        plt.title("Fourier Transform(Log-Log Scale)")
        plt.loglog(xf, np.abs(yf))  # plotting include normalization
        plt.xlabel("Frequency")
        plt.ylabel("Amplitude")

        # plot fourier transform in linear scale
        plt.subplot(3, 1, 3)
        plt.title("Fourier Transform(Linear Scale)")
        plt.plot(xf, np.abs(yf))
        plt.xlabel("Frequency")
        plt.ylabel("Amplitude")

        plt.tight_layout()
        plt.show()

        # Plot original time series
        # plt.figure()
        # plt.subplot(2, 1, 1)
        # plt.title("Original Time Series")
        # plt.plot(self.get_time_raw(), force_data)
        # plt.xlabel("Time")
        # plt.ylabel(force)
        # # plot welch method
        # plt.subplot(2, 1, 2)
        # plt.title("Welch Method")
        # plt.loglog(f, Pxx_den)
        # plt.xlabel("Frequency")
        # plt.ylabel("PSD")

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


class PostPlots:
    def __init__(self, cfd_df, exp_df, exp_df_std):
        self.cfd_fin_df = cfd_df[0]
        self.cfd_flap_df = cfd_df[1]
        self.exp_flap_df = exp_df[1]
        self.exp_fin_df = exp_df[0]
        self.exp_flap_df_std = exp_df_std[1]
        self.exp_fin_df_std = exp_df_std[0]

    def dict_search(self, force):
        force_dict = {
            "moment": "CM",
            "lift": "CL",
            "drag": "CD",
        }
        return force_dict[force]

    def plot_aoa(self, aoa, force):
        # data sort
        fin_cfd = self.cfd_fin_df.query(f"aoa == {str(aoa)}")
        fin_exp_df_data = self.exp_fin_df.query(f"aoa == {str(aoa)}")
        fin_exp_df_std = self.exp_fin_df_std.query(f"aoa == {str(aoa)}")

        flap_cfd = self.cfd_flap_df.query(f"aoa == {str(aoa)}")
        flap_exp_df_data = self.exp_flap_df.query(f"aoa == {str(aoa)}")
        flap_exp_df_std = self.exp_flap_df_std.query(f"aoa == {str(aoa)}")

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].scatter(
            fin_cfd["def"],
            fin_cfd[self.dict_search(force)],
            marker="o",
            c="r",
            label=f"alpha={aoa}(CFD)",
        )
        axs[0].scatter(
            fin_exp_df_data["def"],
            fin_exp_df_data[force],
            marker="s",
            c="k",
            label=f"alpha={aoa}(Exp)",
        )
        axs[0].errorbar(
            fin_exp_df_data["def"],
            fin_exp_df_data[force],
            yerr=fin_exp_df_std[force],
            fmt="none",
            c="k",
            capsize=5,
        )
        axs[0].set_title("Fin")
        axs[1].set_title("Flap")
        axs[1].scatter(
            flap_cfd["def"],
            flap_cfd[self.dict_search(force)],
            marker="o",
            c="r",
            label=f"alpha={aoa}(CFD)",
        )
        axs[1].scatter(
            flap_exp_df_data["def"],
            flap_exp_df_data[force],
            marker="s",
            c="k",
            label=f"alpha={aoa}(Exp)",
        )
        axs[1].errorbar(
            flap_exp_df_data["def"],
            flap_exp_df_data[force],
            yerr=flap_exp_df_std[force],
            fmt="none",
            c="k",
            capsize=5,
        )
        for ax in axs:
            ax.set_xlabel("Deflection Angle (deg)")
            ax.set_ylabel(f"{force.capitalize()} Coefficient")
            ax.set_ylim(0, 0.3)
            ax.legend()
        fig.tight_layout()
        plt.show(block=False)


# functions


def force_coef_std_calc(dict, force, force_std):
    rho, rho_std = dict["rho"]
    velocity, velocity_std = dict["velocity"]
    diameter, diameter_std = dict["diameter"]
    force_coef = force / (0.5 * rho * velocity**2 * ((np.pi * diameter**2 / 4)))
    force_coef_std = force_coef * np.sqrt(
        (force_std / force) ** 2
        + (rho_std / rho) ** 2
        + (4 * velocity_std / velocity) ** 2
        + (4 * diameter_std / diameter) ** 2
    )
    return force_coef, force_coef_std


def force_repeat_avg_calc(force_list, force_std_list):
    force_avg = np.mean(force_list)
    force_std_avg = (
        np.sqrt(np.sum(np.array(force_std_list) ** 2))
        / np.sum(np.array(force_list))
        * force_avg
    )
    return force_avg, force_std_avg


def compare_avg_plot(*args, **kwargs):
    for key, value in kwargs.items():
        if key == "force":
            force = value
        elif key == "title":
            title = value
        elif key == "aoa":
            aoa = value
    # Create a figure and a grid of subplots
    n_instances = len(args)
    fig, axs = plt.subplots(n_instances, 1, figsize=(15, 5))

    for i, obj in enumerate(args):
        y_label = f"{force[i].capitalize()} Force (N)"
        x_label = "Time(S)"
        dict = {"moment": 3, "lift": 1, "drag": 2}
        column = dict[force[i]]
        min_time, max_time = {
            1: (0.0495, 0.0645) if aoa[i] != 0 else (0.05, 0.07),
            2: (0.0460, 0.0645),
            3: (0.0460, 0.0645),
        }[column]
        obj.get_avg(force[i])
        if not isinstance(obj, ExpPostProcess):
            print(f"Skipping argument {i+1}, not a ExpPostProcess object.")
            continue

        if n_instances == 1:  # Special case when only one subplot
            ax = axs
        else:
            ax = axs[i]

        ax.plot(
            obj.df.iloc[:, 0],
            obj.df.iloc[:, column],
            label="Original Data",
            color="blue",
        )
        ax.plot(
            obj.df.iloc[:, 0],
            obj.df[f"smooth {dict[force[i]]}"],
            color="red",
            label="Moving Average",
        )
        ax.set_title(f"{title[i]}")
        ax.axvline(x=min_time, color="black", linestyle="--")
        ax.axvline(x=max_time, color="black", linestyle="--")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        # Add major grid lines
        ax.grid(which="major", linestyle="-", linewidth="0.5", color="black")

        # Add minor grid lines
        ax.minorticks_on()
        ax.grid(which="minor", linestyle=":", linewidth="0.5", color="gray")
    plt.tight_layout()
    plt.show(block=False)


class PostPlots_bcone(PostPlots):
    def __init__(self, cfd_df, exp_df, exp_df_std):
        self.cfd_df = cfd_df
        self.exp_df = exp_df
        self.exp_df_std = exp_df_std

    def plot_aoa(self, force):
        # data sort
        bcone_cfd = self.cfd_df
        bcone_df_data = self.exp_df
        bcone_df_std = self.exp_df_std

        fig, axs = plt.subplots(figsize=(12, 6))
        axs.scatter(
            bcone_cfd["aoa"],
            bcone_cfd[self.dict_search(force)],
            marker="o",
            c="r",
            label=f"CFD",
        )
        axs.scatter(
            bcone_df_data["aoa"],
            bcone_df_data[force],
            marker="s",
            c="k",
            label=f"Exp",
        )
        axs.errorbar(
            bcone_df_data["aoa"],
            bcone_df_data[force],
            yerr=bcone_df_std[force],
            fmt="none",
            c="k",
            capsize=5,
        )

        axs.set_xlabel("Deflection Angle (deg)")
        axs.set_ylabel(f"{force.capitalize()} Coefficient")
        axs.set_ylim(0, 0.3)
        axs.legend()
        fig.tight_layout()
        plt.show()


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
    rho_std = (7.1 / 100) * rho  # in percent
    velocity = 1553
    velocity_std = 1.6 / 100 * velocity  # in percent
    diameter = 0.06
    diameter_std = 0.0005
    # put all conditions in a dictionary
    condition_dict = {
        "rho": (rho, rho_std),
        "velocity": (velocity, velocity_std),
        "diameter": (diameter, diameter_std),
    }
    # using moving average to smooth data
    # test = ExpPostProcess("TR009.csv")
    # test.plot_avg("lift")
    # for bicone_key in [0, 2, 4, 6]:
    #     if type(bicone_data[bicone_key]) == list:
    #         for file in bicone_data[bicone_key]:
    #             print(file)
    #             test = ExpPostProcess(file)
    #             test.plot_avg("lift")
    #     else:
    #         test = ExpPostProcess(bicone_data[bicone_key])
    #         test.plot_avg("lift")
    # test.find_fft("lift")
    fins = [ExpPostProcess(file, 0) for file in fin_config_data[0].values()]
    flaps = [ExpPostProcess(file, 0) for file in flap_config_data[0].values()]
    title_fin = ["Fin, AoA = 0, Delta = 0", "Fin, AoA = 0, Delta = 10"]
    title_flap = ["Flap, AoA = 0, Delta = 5", "Flap, AoA = 0, Delta = 17.5"]
    compare_avg_plot(*fins, force=["lift", "lift"], title=title_fin, aoa=[0, 0])
    compare_avg_plot(*flaps, force=["lift", "lift"], title=title_flap, aoa=[0, 0])

    # %% processing repeats
    # bicone
    bicone_exp = {}
    bicone_repeats = {}

    for key in [0, 2, 4, 6]:
        if type(bicone_data[key]) == list:
            bicone_exp[key] = {}
            bicone_repeats = {}
            for key_force in ["drag", "lift", "moment"]:
                force_list = []
                force_std_list = []
                for file in bicone_data[key]:
                    test = ExpPostProcess(file, key)
                    force, force_std = test.get_avg(key_force)
                    force_list.append(force)
                    force_std_list.append(force_std)
                force_avg, force_std_avg = force_repeat_avg_calc(
                    force_list, force_std_list
                )
                force_coef, force_coef_std = force_coef_std_calc(
                    condition_dict, force_avg, force_std_avg
                )
                bicone_repeats[key_force] = (force_coef, force_coef_std)
            bicone_exp[key] = bicone_repeats
        else:
            # if its not a list
            bicone_exp[key] = {}
            for key_force in ["drag", "lift", "moment"]:
                test = ExpPostProcess(bicone_data[key], key)
                force, force_std = test.get_avg(key_force)
                force_coef, force_coef_std = force_coef_std_calc(
                    condition_dict, force, force_std
                )
                bicone_exp[key][key_force] = (force_coef, force_coef_std)

    # fin config
    fin_exp = {}
    fin_config_repeats = {}

    for key in [0, 5]:
        fin_exp[key] = {}
        for key_def in [0, 10]:
            if type(fin_config_data[key][key_def]) == list:
                for key_force in ["drag", "lift", "moment"]:
                    force_list = []
                    force_std_list = []
                    for file in fin_config_data[key][key_def]:
                        test = ExpPostProcess(file, key)
                        force, force_std = test.get_avg(key_force)
                        force_list.append(force)
                        force_std_list.append(force_std)
                    force_avg, force_std_avg = force_repeat_avg_calc(
                        force_list, force_std_list
                    )
                    force_coef, force_coef_std = force_coef_std_calc(
                        condition_dict, force_avg, force_std_avg
                    )
                    fin_config_repeats[key_force] = (force_coef, force_coef_std)
                fin_exp[key][key_def] = fin_config_repeats
            else:
                # if its not a list
                fin_exp[key][key_def] = {}
                for key_force in ["drag", "lift", "moment"]:
                    test = ExpPostProcess(fin_config_data[key][key_def], key)
                    force, force_std = test.get_avg(key_force)
                    force_coef, force_coef_std = force_coef_std_calc(
                        condition_dict, force, force_std
                    )
                    fin_exp[key][key_def][key_force] = (force_coef, force_coef_std)

    # flap config
    flap_exp = {}
    flap_config_repeats = {}
    for key in [0, 5]:
        flap_exp[key] = {}
        for key_def in [5, 17.5]:
            if type(flap_config_data[key][key_def]) == list:
                for key_force in ["drag", "lift", "moment"]:
                    force_list = []
                    force_std_list = []
                    for file in flap_config_data[key][key_def]:
                        test = ExpPostProcess(file, key)
                        force, force_std = test.get_avg(key_force)
                        force_list.append(force)
                        force_std_list.append(force_std)
                    force_avg, force_std_avg = force_repeat_avg_calc(
                        force_list, force_std_list
                    )
                    force_coef, force_coef_std = force_coef_std_calc(
                        condition_dict, force_avg, force_std_avg
                    )
                    flap_config_repeats[key_force] = (force_coef, force_coef_std)
                flap_exp[key][key_def] = flap_config_repeats
            else:
                # if its not a list
                flap_exp[key][key_def] = {}
                for key_force in ["drag", "lift", "moment"]:
                    test = ExpPostProcess(flap_config_data[key][key_def], key)
                    force, force_std = test.get_avg(key_force)
                    force_coef, force_coef_std = force_coef_std_calc(
                        condition_dict, force, force_std
                    )
                    flap_exp[key][key_def][key_force] = (force_coef, force_coef_std)

    # %% reading ods for cfd data
    os.chdir("data")
    bicone_cfd = pd.read_excel("combined.ods", engine="odf", sheet_name="d_cone_cfd")
    fin_cfd = pd.read_excel("combined.ods", engine="odf", sheet_name="fin_cfd")
    flap_cfd = pd.read_excel("combined.ods", engine="odf", sheet_name="flap_cfd")
    print(bicone_cfd)
    # %% plotting with cfd and exp

    # flatten bicone
    record = []
    for aoa, aoa_data in bicone_exp.items():
        for force_typ, force_data in aoa_data.items():
            record_dict = {
                "aoa": aoa,
                "force": force_typ,
                "force_coef": force_data[0],
                "force_coef_std": force_data[1],
            }
            record.append(record_dict)
    bicone_exp_df = pd.DataFrame(record)
    bicone_exp_df_data = pd.pivot_table(
        bicone_exp_df, values="force_coef", index=["aoa"], columns=["force"]
    ).reset_index()
    bicone_exp_df_std = pd.pivot_table(
        bicone_exp_df, values="force_coef_std", index=["aoa"], columns=["force"]
    ).reset_index()

    # plot fin and flap side by side
    # flatted fin dictionary
    record = []
    record_flap = []
    for aoa, aoa_data in fin_exp.items():
        for defl, defl_data in aoa_data.items():
            for force_typ, force_data in defl_data.items():
                record_dict = {
                    "aoa": aoa,
                    "def": defl,
                    "force": force_typ,
                    "force_coef": force_data[0],
                    "force_coef_std": force_data[1],
                }
                record.append(record_dict)
                if aoa in [0, 5] and defl == 0:
                    record_flap.append(record_dict)

    fin_exp_df = pd.DataFrame(record)
    fin_exp_df_data = pd.pivot_table(
        fin_exp_df, values="force_coef", index=["aoa", "def"], columns=["force"]
    ).reset_index()
    fin_exp_df_std = pd.pivot_table(
        fin_exp_df, values="force_coef_std", index=["aoa", "def"], columns=["force"]
    ).reset_index()

    # flatted flap dictionary
    for aoa, aoa_data in flap_exp.items():
        for defl, defl_data in aoa_data.items():
            for force_typ, force_data in defl_data.items():
                record_dict = {
                    "aoa": aoa,
                    "def": defl,
                    "force": force_typ,
                    "force_coef": force_data[0],
                    "force_coef_std": force_data[1],
                }
                record_flap.append(record_dict)
    flap_exp_df = pd.DataFrame(record_flap)
    flap_exp_df_data = pd.pivot_table(
        flap_exp_df, values="force_coef", index=["aoa", "def"], columns=["force"]
    ).reset_index()
    flap_exp_df_std = pd.pivot_table(
        flap_exp_df, values="force_coef_std", index=["aoa", "def"], columns=["force"]
    ).reset_index()

    temp = PostPlots(
        [fin_cfd, flap_cfd],
        [fin_exp_df_data, flap_exp_df_data],
        [fin_exp_df_std, flap_exp_df_std],
    )
    temp.plot_aoa(0, "drag")
    temp.plot_aoa(5, "drag")
    temp.plot_aoa(0, "lift")
    temp.plot_aoa(5, "lift")

    bcone = PostPlots_bcone(bicone_cfd, bicone_exp_df_data, bicone_exp_df_std)
    bcone.plot_aoa("drag")
    bcone.plot_aoa("lift")
