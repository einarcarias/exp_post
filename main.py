import os
import random
import re
from pathlib import Path
from typing import Dict, List, Union

import cmasher as cmr
import matplotlib as mpl
import numpy as np
import pandas as pd
import scienceplots
import sympy as sym
import tikzplotlib
from matplotlib import pyplot as plt
from matplotlib.legend import Legend
import sympy as sp

# monkey patching
from matplotlib.lines import Line2D
from numpy import cos, pi, sin, sqrt
from scipy import signal
from scipy.fft import fft, fftfreq, ifft
from scipy.interpolate import interp1d as interp
from scipy.signal import butter, filtfilt, welch
from statsmodels.tsa.seasonal import STL

Line2D._us_dashSeq = property(lambda self: self._dash_pattern[1])
Line2D._us_dashOffset = property(lambda self: self._dash_pattern[0])
Legend._ncol = property(lambda self: self._ncols)
plt.style.use(["science", "bright"])
plt.rcParams.update(
    {
        "font.family": "serif",  # specify font family here
        "font.serif": ["Times"],  # specify font here
        "font.size": 11,
    }
)
dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = Path("data")
# default dpi 300 when saving
plt.rcParams["savefig.dpi"] = 300


class ExpPostProcess:
    data_dir = Path("data")

    def __init__(
        self,
        filename,
        config,
        aoa,
        deflection,
        label=None,
    ):
        os.chdir(self.data_dir)
        # extract numeric portion of config data name to determine test number
        self.config = config.capitalize()
        self.aoa = aoa
        self.deflection = deflection
        self.filename = filename
        # determine if file has .xlsx or .csv extension
        self.df = self.xlsx_or_csv(filename)
        self.test_type()
        self.sampling_rate = self.get_sampling_rate()
        self.averaging_window = 60 if self.sampling_rate < 10006 else 200
        self.label = self.name() if label is None else label

        os.chdir(dir_path)

    # determine if test is bicone background tap or fin, flap
    def test_type(self):
        if self.config not in ["Fin", "Flap"]:
            self.test_number = None
        else:
            self.test_number = int(re.search(r"(\d+)", self.filename).group(1))

    # getters for raw data
    def get_moment_raw(self):
        return self.df.iloc[:, 3]

    def get_normal_raw(self):
        return self.df.iloc[:, 1]

    def get_time_raw(self):
        return self.df.iloc[:, 0]

    def get_sampling_rate(self):
        time = self.get_time_raw()
        return int(len(time) / (time.iloc[-1] - time.iloc[0]))

    def get_axial_raw(self):
        return self.df.iloc[:, 2]

    def get_minmax(self):
        file_loc = self.data_dir / "exp_time_avg.csv"
        pd_time_avg = pd.read_csv(file_loc, dtype=np.float64, index_col=0)
        test_num = self.test_number
        min_time = pd_time_avg.loc[test_num]["Time min"]
        max_time = pd_time_avg.loc[test_num]["Time max"]
        return min_time, max_time

    def dict_search(self, force):
        force_dict = {
            "moment": self.get_moment_raw(),
            "normal": self.get_normal_raw(),
            "axial": self.get_axial_raw(),
        }
        return force_dict[force]

    def plot_raw(self, force, save_name=None, to_tikz=False):
        force_name = force
        force = self.dict_search(force)
        time = self.get_time_raw()
        force_avg = force.rolling(window=self.averaging_window, center=True).mean()
        plt.figure(figsize=(6, 3))
        plt.plot(time, force)
        plt.plot(time, force_avg, color="red")
        plt.xlabel(bold_text("Time(s)"))
        plt.ylabel(
            bold_text(
                f"{force_name.capitalize()} Force ({'N' if force_name in ['normal','axial'] else 'Nm'})"
            )
        )
        plt.xlim(0, 0.2)
        plt.tight_layout()
        if to_tikz and (save_name is not None):
            tikzplotlib.clean_figure()
            tikzplotlib.save(
                f"{save_name}.tex",
                axis_width=".8\\textwidth",
                axis_height=".8*\\axisdefaultheight",
                table_row_sep=r"\\ ",
            )
        else:
            plt.show()
    # static method
    @staticmethod
    def xlsx_or_csv(filename):
        if ".xlsx" in filename:
            return pd.read_excel(filename, dtype=np.float64)
        else:
            return pd.read_csv(filename, delimiter="\t", dtype=np.float64)

    def plot_avg(
        self, force, use_cutoff=False, y_lim=None, manual_cutoff=(), to_tikz=False
    ):
        dict = {"moment": 3, "normal": 1, "axial": 2}
        column = dict[force]
        self.get_avg(force, manual_cutoff)
        if manual_cutoff != ():
            min_time, max_time = manual_cutoff
        else:
            min_time, max_time = self.get_minmax()

        # Plotting original and smoothed data
        plt.figure(figsize=(6, 3))
        plt.plot(
            self.df.iloc[:, 0],
            self.df.iloc[:, column],
            label="Original Data",
        )
        plt.plot(
            self.df.iloc[:, 0],
            self.df[f"smooth {dict[force]}"],
            label=f"Moving Average",
            color="red",
        )

        plt.axvline(x=min_time, color="black", linestyle="--")
        plt.axvline(x=max_time, color="black", linestyle="--")
        plt.xlabel(bold_text("Time(s)"))
        plt.ylabel(
            bold_text(
                f"{force.capitalize()} {'Force (N' if force in ['normal','axial'] else '(Nm'})"
            )
        )
        # plt.title(f"Centered Moving Average Smoothing for {self.name()}")
        # add white background to legend

        legend = plt.legend(
            loc="lower right", frameon=1, facecolor="white", framealpha=1
        )
        frame = legend.get_frame()
        frame.set_linewidth(0.0)
        min_lim = min_time if use_cutoff else 0
        max_lim = max_time if use_cutoff else 0.2
        plt.xlim(min_lim, max_lim)
        if y_lim is not None:
            plt.ylim(y_lim[0], y_lim[1])
        plt.tight_layout()
        if to_tikz:
            name = (
                f"time_{self.config.lower()}a{self.aoa}d{self.deflection}_40k_{force}"
            )
            tikzplotlib.clean_figure()
            tikzplotlib.save(
                f"{name}.tex",
                axis_width=".8\\textwidth",
                axis_height=".8*\\axisdefaultheight",
                table_row_sep=r"\\ ",
            )
        else:
            plt.show()

    def name(self):
        return rf"{self.config}, $\alpha$={self.aoa}, $\delta$ = {self.deflection}"

    def get_avg(self, force, manual_cutoff=()):
        dict = {"moment": 3, "normal": 1, "axial": 2}
        column = dict[force]
        window = self.averaging_window
        smooth = self.df.iloc[:, column].rolling(window=window, center=True).mean()
        smooth.fillna(0, inplace=True)
        self.df[f"smooth {dict[force]}"] = smooth
        if manual_cutoff != ():
            min_time, max_time = manual_cutoff
        else:
            min_time, max_time = self.get_minmax()
        relevant_data = (self.df.iloc[:, 0] >= min_time) & (
            self.df.iloc[:, 0] <= max_time
        )
        indices = self.df[relevant_data].index
        std = self.df.loc[indices, [f"smooth {dict[force]}"]].std()
        avg_force = self.df.loc[indices, [f"smooth {dict[force]}"]].mean()

        return avg_force.iloc[0], std.iloc[0]

    def cut_data(self, start_time, end_time):
        # reset index
        time = self.get_time_raw()
        start_index = time[time >= start_time].index[0]
        end_index = time[time <= end_time].index[-1]
        self.df = self.df.iloc[start_index : end_index + 1, :]
        self.df.reset_index(drop=True, inplace=True)

    # post proccessing methods starts here

    def find_welch(self, force):
        force_data = self.dict_search(force)
        b = 5
        f, Pxx_den = welch(
            force_data - force_data.mean(),
            fs=self.sampling_rate,
            nperseg=self.sampling_rate // b,
        )

        return pd.DataFrame(
            {"freq": f, "psd": Pxx_den, "label": self.label, "config": self.config}
        )

    def rms(self, force):
        force_data = self.dict_search(force)
        time = self.get_time_raw()
        window = self.averaging_window
        zero_mean = force_data.pow(2).rolling(window, center=True).std(ddof=0)
        zero_mean.fillna(0, inplace=True)
        smooth = (
            force_data.pow(2).rolling(window=window, center=True).mean().apply(np.sqrt)
        )
        smooth_rolling_av = force_data.rolling(window=window, center=True).mean()
        smooth_rolling_av.fillna(0, inplace=True)
        min_time, max_time = self.get_minmax()
        # plot
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(time, force_data, c="b", label="Original")
        ax.plot(time, smooth, c="r", label="Smooth")
        # ax.plot(time, zero_mean, c="g", label="Zero Mean")
        ax.plot(time, smooth_rolling_av, c="k", label="Smooth Rolling Average")
        ax.legend()
        plt.xlabel(bold_text("Time(s)"))
        plt.ylabel(
            bold_text(
                f"{force.capitalize()} Force ({'N' if force in ['lift','drag'] else 'Nm'})"
            )
        )
        plt.xlim(0, 0.2)
        plt.tight_layout()
        plt.axvline(x=min_time, color="black", linestyle="--")
        plt.axvline(x=max_time, color="black", linestyle="--")
        plt.show()

    @staticmethod
    def plot_freq_domain(data, xmin=None, xmax=None):
        cmap = mpl.colormaps["viridis"]
        fig, ax = plt.subplots(figsize=(10, 4))
        for i in data:
            if i["config"][0] in ["Background", "Tap 1", "Tap 2"]:
                lwidth = 1.5
                (line,) = ax.loglog(
                    i["freq"],
                    i["psd"],
                    linewidth=lwidth,
                    label=i["label"][0],
                    c=(
                        "k"
                        if i["config"][0] == "Background"
                        else "g" if i["config"][0] == "Tap 1" else "orange"
                    ),
                )
            else:
                lwidth = 1
                ax.loglog(
                    i["freq"],
                    i["psd"],
                    linewidth=lwidth,
                    label=i["label"][0],
                    alpha=0.4,
                )
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.minorticks_on()
        ax.grid(which="minor", linestyle=":", linewidth="0.5", color="gray")
        # ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=len(data) / 4)
        ax.set_ylim(0, 2e2)
        if xmin is not None:
            fig.set_size_inches(5, 4)
            ax.set_xlim(xmin, xmax)
            ax.xaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())
        plt.xlabel(bold_text("Frequency, Hz"))
        plt.ylabel(bold_text(r"PSD, N$\mathrm{^2}$/Hz"))
        plt.tight_layout()

        # add legend outside the plot window below

        plt.show(block=True)


class ExpPostProcess_past(ExpPostProcess):
    def __init__(
        self,
        filename,
        config,
        aoa,
        deflection,
        label=None,
    ):
        super().__init__(filename, config, aoa, deflection, label=None)

    # getters for raw data

    def get_axial_raw(self):
        return self.df.iloc[:, 1]

    def get_time_raw(self):
        return self.df.iloc[:, 0]

    def get_moment_raw(self):
        print("Moment data not available for this test.")

    def get_normal_raw(self):
        print("Normal data not available for this test.")

    def get_time_raw(self):
        return self.df.iloc[:, 0]

    @staticmethod
    def plot_freq_domain(data, xmin=None, xmax=None):
        cmap = mpl.colormaps["viridis"]
        fig, ax = plt.subplots(figsize=(10, 4))
        lwidth = 1
        ax.loglog(
            data["freq"],
            data["psd"],
            linewidth=lwidth,
            label="test",
            alpha=0.4,
        )
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.minorticks_on()
        ax.grid(which="minor", linestyle=":", linewidth="0.5", color="gray")
        # ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=len(data) / 4)
        ax.set_ylim(0, 2e2)
        if xmin is not None:
            fig.set_size_inches(5, 4)
            ax.set_xlim(xmin, xmax)
            ax.xaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())
        plt.xlabel(bold_text("Frequency, Hz"))
        plt.ylabel(bold_text(r"PSD, N$\mathrm{^2}$/Hz"))
        plt.tight_layout()

        # add legend outside the plot window below

        plt.show()

    def plot_avg(self, force):
        dict = {"moment": 3, "normal": 1, "axial": 2}
        column = dict[force]
        self.get_avg(force)
        min_time, max_time = self.get_minmax()

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
            label=f"Moving Average",
            color="red",
        )

        plt.axvline(x=min_time, color="black", linestyle="--")
        plt.axvline(x=max_time, color="black", linestyle="--")
        plt.xlabel(bold_text("Time(s)"))
        plt.ylabel(
            bold_text(
                f"{force.capitalize()} {'Force (N' if force in ['normal','axial'] else '(Nm'})"
            )
        )
        # plt.title(f"Centered Moving Average Smoothing for {self.name()}")
        # add white background to legend

        legend = plt.legend(
            loc="lower right", frameon=1, facecolor="white", framealpha=1
        )
        frame = legend.get_frame()
        frame.set_linewidth(0.0)
        min_lim = min_time if use_cutoff else 0
        max_lim = max_time if use_cutoff else 0.2
        plt.xlim(min_lim, max_lim)
        if y_lim is not None:
            plt.ylim(y_lim[0], y_lim[1])
        plt.tight_layout()
        plt.show()


class PostPlots:
    """
    A class for creating scatter plots with error bars for CFD and experimental data.

    Args:
        cfd_df (pd.DataFrame): A DataFrame containing CFD data.
        exp_df (pd.DataFrame): A DataFrame containing experimental data.

    Attributes:
        cfd_df (pd.DataFrame): A DataFrame containing CFD data.
        exp_df (pd.DataFrame): A DataFrame containing experimental data.
    """

    def __init__(
        self, cfd_df: pd.DataFrame, exp_df: pd.DataFrame, invis_df: pd.DataFrame
    ):
        self.cfd_df = cfd_df
        self.exp_df = exp_df
        self.invis_df = invis_df

    def dict_search(self, force: str) -> str:
        """
        Search for the corresponding force abbreviation based on the force name.

        Args:
            force (str): The name of the force ("moment", "lift", or "drag").

        Returns:
            str: The corresponding force abbreviation ("CM", "CL", or "CD").
        """
        force_dict: Dict[str, str] = {
            "moment": "CM",
            "lift": "CL",
            "drag": "CD",
        }
        return force_dict[force]

    def plot_aoa(
        self, aoa_values: float, force: str, config: str, to_tikz=False
    ) -> None:
        """
        Plot scatter plots with error bars for angle of attack (aoa) value(s)
        and a specified force type.

        Args:
            aoa_values (Union[float, List[float]]): A single angle of attack (aoa) value or a list of aoa values.
            force (str): The force type ("moment", "lift", or "drag").

        Returns:
            None
        """
        force_target: str = self.dict_search(force)

        exp_df_no_tuple = self.exp_df.copy()
        for force_i in ["CL", "CD", "CM"]:
            exp_df_no_tuple[force_i] = exp_df_no_tuple[force_i].apply(lambda x: x[0])
        concat_df = pd.concat([self.cfd_df, self.invis_df, exp_df_no_tuple])
        exp_max_std = self.exp_df[force_target].apply(lambda x: x[1]).max()

        # Ensure aoa_values is a list

        max_val = {"CL": 0.38, "CD": 0.25, "CM": 0.2}[force_target]

        cfd_error  = np.abs(8.11-8.2)

        fig, axs = plt.subplots(figsize=(5, 4))

        # Data sort
        cfd = self.cfd_df.query(f"aoa == {str(aoa_values)}")
        exp = self.exp_df.query(f"aoa == {str(aoa_values)}")
        invis = self.invis_df.query(f"aoa == {str(aoa_values)}")

        (cfd_plot,) = axs.plot(
            cfd["deflection"],
            cfd[force_target],
            marker="^",
            linestyle="--",
            label=rf"CFD",
        )
        print(f"CFD plot color {cfd_plot.get_color()}")
        (code_plot,) = axs.plot(
            invis["deflection"],
            invis[force_target],
            marker="None",
            linestyle="-",
            linewidth=1.5,
            label=f"Inviscid Code",
        )
        print(f"Inviscid code plot color {code_plot.get_color()}")

        axs.errorbar(
            exp["deflection"],
            exp[force_target].apply(lambda x: x[0]),
            yerr=exp[force_target].apply(lambda x: x[1]),
            fmt="none",
            c="k",
            capsize=5,
        )
        axs.plot(
            exp["deflection"],
            exp[force_target].apply(lambda x: x[0]),
            marker="s",
            linestyle="None",
            c="k",
            label="Experiment",
        )

        axs.set_xlabel(bold_text("Deflection Angle (deg)"))
        axs.set_ylabel(bold_text(f"{force.capitalize()} Coefficient"))
        plt.ylim([0.0, max_val])
        plt.xlim([-0.1, cfd["deflection"].max() + 0.5])
        axs.legend(loc="best")
        fig.tight_layout()
        # tikzplotlib.clean_figure(fig)
        if to_tikz:
            tikzplotlib.save(
                f"{force}_aoa{aoa_values}_comparison_{config}.tex", table_row_sep="\\\\"
            )
        else:
            plt.show()
        # plt.show(block=True)


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
        fig.show()
        fig.tight_layout()


# class for error analysis
class ErrorAnalysis:
    def __init__(self, ExpPostProcess_object, boundary_conditions):
        if not isinstance(ExpPostProcess_object, ExpPostProcess):
            raise TypeError("ExpPostProcess object required.")
        else:
            self.ExpPostProcess_object = ExpPostProcess_object
        self.rho, self.rho_std = boundary_conditions["rho"]
        self.velocity, self.velocity_std = boundary_conditions["velocity"]
        self.diameter, self.diameter_std = boundary_conditions["diameter"]

    def coef_calc(self, force, manual_cutoff=()):
        # obtain normal and axial forces std and avg
        aoa_std = 0.1
        if manual_cutoff != ():
            normal_avg, normal_std = self.ExpPostProcess_object.get_avg(
                "normal", manual_cutoff
            )
            axial_avg, axial_std = self.ExpPostProcess_object.get_avg(
                "axial", manual_cutoff
            )
        else:
            normal_avg, normal_std = self.ExpPostProcess_object.get_avg("normal")
            axial_avg, axial_std = self.ExpPostProcess_object.get_avg("axial")

        def calculate_drag_avg_std(
            avg_normal, std_normal, avg_axial, std_axial, avg_alpha, std_alpha, k
        ):
            # Convert angle from degrees to radians
            avg_alpha_rad = np.deg2rad(avg_alpha)
            std_alpha_rad = np.deg2rad(std_alpha)

            # Calculate average lift using the respective means of normal, axial coefficients, and alpha
            avg_drag = cd_calc(avg_axial, avg_normal, avg_alpha_rad)
            # Calculate variance of lift using the standard deviations assuming independence
            k_n = k[0]
            k_n = k_n.subs(sp.symbols("n"), avg_normal)
            k_n = k_n.subs(sp.symbols("a"), avg_axial)
            k_n = k_n.subs(sp.symbols("alpha"), avg_alpha_rad)
            k_a = k[1]
            k_a = k_a.subs(sp.symbols("n"), avg_normal)
            k_a = k_a.subs(sp.symbols("a"), avg_axial)
            k_a = k_a.subs(sp.symbols("alpha"), avg_alpha_rad)
            k_alpha = k[2]
            k_alpha = k_alpha.subs(sp.symbols("n"), avg_normal)
            k_alpha = k_alpha.subs(sp.symbols("a"), avg_axial)
            k_alpha = k_alpha.subs(sp.symbols("alpha"), avg_alpha_rad)

            alpha_calc = 0 if avg_alpha == 0 else (float(k_alpha)*std_alpha_rad / avg_alpha_rad) ** 2
            var_drag = sqrt(
                (float(k_a)*std_axial / avg_axial) ** 2
                + (float(k_n)*std_normal / avg_normal) ** 2
                + alpha_calc
            )

            # Convert variance to standard deviation
            std_drag = var_drag * avg_drag

            return avg_drag, std_drag

        def calculate_lift_avg_std(
            avg_normal, std_normal, avg_axial, std_axial, avg_alpha, std_alpha, k
        ):
            # Convert angle from degrees to radians
            avg_alpha_rad = np.deg2rad(avg_alpha)
            std_alpha_rad = np.deg2rad(std_alpha)

            # Calculate average drag using the respective means of normal, axial coefficients, and alpha
            avg_lift = cl_calc(avg_axial, avg_normal, avg_alpha_rad)
            # Calculate the variance of the lift coefficient assuming statistical independence
            k_n = k[0]
            # sub in values
            k_n = k_n.subs(sp.symbols("n"), avg_normal)
            k_n = k_n.subs(sp.symbols("a"), avg_axial)
            k_n = k_n.subs(sp.symbols("alpha"), avg_alpha_rad)
            k_a = k[1]
            k_a = k_a.subs(sp.symbols("n"), avg_normal)
            k_a = k_a.subs(sp.symbols("a"), avg_axial)
            k_a = k_a.subs(sp.symbols("alpha"), avg_alpha_rad)
            k_alpha = k[2]
            k_alpha = k_alpha.subs(sp.symbols("n"), avg_normal)
            k_alpha = k_alpha.subs(sp.symbols("a"), avg_axial)
            k_alpha = k_alpha.subs(sp.symbols("alpha"), avg_alpha_rad)

            alpha_calc = (
                0
                if avg_alpha == 0
                else (float(k_alpha) * std_alpha_rad / avg_alpha_rad) ** 2
            )

            var_lift = sqrt(
                (float(k_a) * std_axial / avg_axial) ** 2
                + (float(k_n) * std_normal / avg_normal) ** 2
                + (alpha_calc)
            )

            # Convert variance to standard deviation
            std_lift = var_lift * avg_lift

            return avg_lift, std_lift

        # symbolic equations to find k
        n_sym, a_sym, alpha_sym = sp.symbols("n a alpha")

        # lift equation
        cl_sym = (n_sym * sp.cos(alpha_sym)) - (a_sym * sp.sin(alpha_sym))
        k_l_n = sp.diff(cl_sym, n_sym) * (n_sym / cl_sym)
        k_l_a = sp.diff(cl_sym, a_sym) * (a_sym / cl_sym)
        k_l_alpha = sp.diff(cl_sym, alpha_sym) * (alpha_sym / cl_sym)
        # print simplified k
        k_l = [k_l_n, k_l_a, k_l_alpha]
        # drag equation
        cd_sym = (n_sym * sp.sin(alpha_sym)) + (a_sym * sp.cos(alpha_sym))
        k_d_n = sp.diff(cd_sym, n_sym) * (n_sym / cd_sym)
        k_d_a = sp.diff(cd_sym, a_sym) * (a_sym / cd_sym)
        k_d_alpha = sp.diff(cd_sym, alpha_sym) * (alpha_sym / cd_sym)
        # print simplified k
        k_d = [k_d_n, k_d_a, k_d_alpha]

        if force == "lift":
            force_avg, force_std = calculate_lift_avg_std(
                normal_avg,
                normal_std,
                axial_avg,
                axial_std,
                self.ExpPostProcess_object.aoa,
                aoa_std,
                k_l,
            )
        elif force == "drag":
            force_avg, force_std = calculate_drag_avg_std(
                normal_avg,
                normal_std,
                axial_avg,
                axial_std,
                self.ExpPostProcess_object.aoa,
                aoa_std,
                k_d,
            )
        else:
            if manual_cutoff != ():
                force_avg, force_std = self.ExpPostProcess_object.get_avg(
                    force, manual_cutoff
                )
            else:
                force_avg, force_std = self.ExpPostProcess_object.get_avg(force)

        force_coef = force_avg / (
            0.5 * self.rho * self.velocity**2 * (np.pi * self.diameter**2 / 4)
            if force in ["lift", "drag"]
            else 0.5
            * self.rho
            * self.velocity**2
            * ((np.pi * self.diameter**2 / 4) * self.diameter)
        )
        term1 = (force_std / force_avg) ** 2
        term2 = (self.rho_std / self.rho) ** 2
        term3 = 4 * ((self.velocity_std / self.velocity) ** 2)
        term4 = (
            4 * ((self.diameter_std / self.diameter) ** 2)
            if force in ["lift", "drag"]
            else 5 * ((self.diameter_std / self.diameter) ** 2)
        )
        coef_std = np.sqrt(term1 + term2 + term3 + term4) * force_coef
        return force_coef.astype(np.float64), coef_std.astype(np.float64)

    @staticmethod
    def repeat_avg(ErrorAnalysisList, force, manual_cutoff=()):
        for i in ErrorAnalysisList:
            if not isinstance(i, ErrorAnalysis):
                raise TypeError("ErrorAnalysis object required.")
        force_list = []
        force_std_list = []
        for i in ErrorAnalysisList:
            if manual_cutoff != ():
                force_list.append(i.coef_calc(force, manual_cutoff)[0])
                force_std_list.append(i.coef_calc(force, manual_cutoff)[1])
            else:
                force_list.append(i.coef_calc(force)[0])
                force_std_list.append(i.coef_calc(force)[1])
        force_avg = np.mean(force_list)
        force_mat = np.array(force_list)
        force_std_mat = np.array(force_std_list)
        force_std_avg = (
            np.sqrt(np.sum(force_std_mat) ** 2) / np.sum(force_mat) * force_avg
        )

        return force_avg, force_std_avg


# %% functions
def cl_calc(axial, normal, aoa):
    return (-axial * np.sin(aoa)) + (normal * np.cos(aoa))


def cd_calc(axial, normal, aoa):
    return (axial * np.cos(aoa)) + (normal * np.sin(aoa))


def process_data_to_dataframe(config_data, condition_dict, config_name, manual_cutoff=()):
    """
    Process and organize the data based on angle of attack (aoa), deflection,
    and force type (drag, lift, moment) and return as a Pandas DataFrame.

    Parameters:
    - fin_config_data (dict): Dictionary containing data for various configurations.
    - condition_dict (dict): Dictionary containing conditions for error analysis.

    Returns:
    - DataFrame: A DataFrame organized by aoa, deflection, force type with processed data.
    """
    # force dictiory
    force_dict = {"moment": "CM", "lift": "CL", "drag": "CD"}

    # List to store processed data
    data_list = []
    diameter_model = condition_dict["diameter"][0]
    diamter_sting = condition_dict["sting_diam"]
    ca_base_0 = BasePressureCalc(condition_dict["mach"]).gabeaud()

    # Loop through each angle of attack (aoa)
    for aoa in config_data.keys():
        # Loop through each deflection angle
        for deflect in config_data[aoa].keys():
            # Loop through each force type: drag, lift, and moment
            for force in ["drag", "lift", "moment"]:
                # Check if the data for a given aoa and deflection is a list

                if type(config_data[aoa][deflect]) == list:
                    force_coef_list = []

                    # Process each file in the list for the current configuration
                    for file in config_data[aoa][deflect]:
                        files = ExpPostProcess(file, config_name, aoa, deflect)
                        coef = ErrorAnalysis(files, condition_dict)
                        force_coef_list.append(coef)

                    # Calculate average and standard deviation of force coefficients
                    #if manual cutoff is not an empty tuple
                    if manual_cutoff != ():
                        force_avg, force_std_avg = ErrorAnalysis.repeat_avg(
                            force_coef_list, force, manual_cutoff
                        )
                    else:
                        force_avg, force_std_avg = ErrorAnalysis.repeat_avg(
                            force_coef_list, force
                        )

                else:
                    # Handle case where data for a given aoa and deflection is not a list
                    file = ExpPostProcess(
                        config_data[aoa][deflect], config_name, aoa, deflect
                    )
                    coef = ErrorAnalysis(file, condition_dict)
                    force_avg, force_std_avg = ErrorAnalysis.repeat_avg([coef], force)

                # Append the processed data to the list

                # if (aoa == 0 and deflect == 0) and (force in ["lift", "moment"]):
                #     force_avg = 0
                data_list.append(
                    [aoa, deflect, force_dict[force], force_avg, force_std_avg]
                )

    # Convert the list to a DataFrame
    df = pd.DataFrame(
        data_list,
        columns=["aoa", "deflection", "force", "avg", "std"],
    )
    # Pivot the DataFrame to have CL, CD, and CM columns
    df_pivot = df.pivot_table(
        index=["aoa", "deflection"], columns="force", values=["avg", "std"]
    )

    # Flatten the column names
    df_pivot.columns = ["_".join(col).strip() for col in df_pivot.columns.values]

    # Reset the index
    df_pivot = df_pivot.reset_index()

    # Rename the columns
    df_pivot = df_pivot.rename(
        columns={
            "avg_CL": "CL",
            "std_CL": "CL_std",
            "avg_CD": "CD",
            "std_CD": "CD_std",
            "avg_CM": "CM",
            "std_CM": "CM_std",
        }
    )

    # Reorder the columns
    df_pivot = df_pivot[
        ["aoa", "deflection", "CL", "CL_std", "CD", "CD_std", "CM", "CM_std"]
    ]

    # Apply base pressure correction to CL and CD columns
    correction = df_pivot.apply(
        lambda row: pd.Series(
            [
                base_pressure_correction(
                    row["CD"],
                    row["CL"],
                    ca_base_0,
                    row["aoa"],
                    np.pi * diameter_model**2 / 4,
                    np.pi * diamter_sting**2 / 4,
                )
            ]
        ),
        axis=1,
    )
    df_pivot["CL"] = correction[0].apply(lambda x: x[0])
    df_pivot["CD"] = correction[0].apply(lambda x: x[1])

    # Convert CL, CD, and CM columns to tuples of (avg, std)
    df_pivot[["CL", "CD", "CM"]] = df_pivot[["CL", "CD", "CM"]].apply(
        lambda col: list(zip(col, df_pivot[col.name + "_std"]))
    )
    df_pivot = df_pivot.drop(columns=["CL_std", "CD_std", "CM_std"])

    return df_pivot


def base_pressure_correction(
    drag_force, lift_force, ca_base_0, alpha, base_area, sting_area
):
    """
    Calculates the corrected lift and drag coefficients based on the base pressure
    coefficient and the measured lift and drag forces.

    Parameters
    ----------
    drag_force : float
        The measured drag force coef.
    lift_force : float
        The measured lift force coef.
    ca_base_0 : float
        The base pressure coefficient at alpha=0.
    alpha : float
        The angle of attack in radians.
    base_area : float
        The area of the base in m^2.
    sting_area : float
        The area of the sting in m^2.

    Returns
    -------
    cl_new : float
        The corrected lift coefficient.
    cd_new : float
        The corrected drag coefficient.
    """
    ratio = sting_area / base_area
    alpha = np.deg2rad(alpha)
    ca_base = ca_base_0[0] if alpha == 0 else ca_base_0[0] * (np.cos(alpha) ** 2)
    ca_base = ca_base * ratio
    ca, cn = sym.symbols("ca,cn")
    eq1 = sym.Eq(cn * sym.cos(alpha) - ca * sym.sin(alpha), lift_force)
    eq2 = sym.Eq(cn * sym.sin(alpha) + ca * sym.cos(alpha), drag_force)
    result = sym.solve([eq1, eq2], (ca, cn))
    cl_new = result[cn] * np.cos(alpha) - (result[ca] + ca_base) * np.sin(alpha)
    cd_new = result[cn] * np.sin(alpha) + (result[ca] + ca_base) * np.cos(alpha)

    return cl_new, cd_new


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
        obj.get_avg(force[i])
        min_time, max_time = obj.cutoff_time_range
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
        # ax.set_title(f"{title[i]}")
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


def bold_text(text):
    return r"\textbf{" + text + "}"


def percent_error(est, true):
    return np.abs(est - true) / (true + 1e-10) * 100


def weighted_mape(y_true, y_pred, std_values):
    """
    Calculate the weighted mean absolute percentage error (wMAPE) between the true and predicted values.

    Args:
        y_true (array-like): The true values.
        y_pred (array-like): The predicted values.
        std_values (array-like): The standard deviation values.

    Returns:
        float: The wMAPE.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    std_values = np.array(std_values)
    weights = 1 / (std_values + 1e-10)  # Adding a small value to avoid division by zero
    absolute_percentage_errors = np.abs((y_true - y_pred) / (y_true + 1e-10)) * 100
    return np.sum(absolute_percentage_errors * weights) / np.sum(weights)


def weighted_mae(y_true, y_pred, std_values):
    """
    Calculate weighted mean absolute error (MAE) based on standard deviations.

    :param y_true: list of true values
    :param y_pred: list of predicted values
    :param std_values: list of standard deviations for the true values
    :return: Weighted MAE
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    weights = 1 / (std_values + 1e-10)  # Adding a small value to avoid division by zero
    absolute_errors = np.abs(y_true - y_pred)
    return np.sum(absolute_errors * weights) / np.sum(weights)


class BasePressureCalc:
    def __init__(self, mach):
        if isinstance(mach, float):
            self.mach_list = [mach]
        elif isinstance(mach, np.ndarray):
            self.mach_list = mach
        else:
            raise TypeError("Mach number must be a float or a list of floats")
        self.gamma = 1.4

    def honeywell(self):
        for mach in self.mach_list:
            if not (0 <= mach < 1.1 or 1.1 <= mach < 5 or mach >= 5):
                raise ValueError("Mach number out of range")

        return [
            (
                -(0.115 + (10 * mach - 3) ** 3 * 10 ** (-4))
                if 0 < mach < 1
                else (
                    -(0.255 - 0.135 * np.log(mach))
                    if 1.1 <= mach < 5
                    else (
                        -(0.019 - 0.012 * (mach - 5) + (0.1 / mach))
                        if mach > 5
                        else None
                    )
                )
            )
            for mach in self.mach_list
        ]

    def gabeaud(self):
        def gab(m):
            term1 = -2 / (self.gamma * m**2)
            term2 = (2 / (self.gamma + 1)) ** self.gamma
            term3 = (1 / m) ** (2 * self.gamma)
            term4 = (2 * self.gamma * m**2 - (self.gamma - 1)) / (self.gamma + 1)
            return term1 * (term2 * term3 * term4 - 1)

        return [-gab(mach) for mach in self.mach_list]

    def love(self):
        os.chdir("data")
        love_df = pd.read_csv("love_exp.csv", header=None, names=["x", "y"])
        os.chdir("..")
        x_data = love_df["x"]
        y_data = love_df["y"]
        coeffs = np.polyfit(x_data, y_data, 3)
        poly = np.poly1d(coeffs)
        interp_func = interp(
            x_data, poly(x_data), kind="cubic", fill_value="extrapolate"
        )
        return interp_func(self.mach_list)


# NOTES: CONSTANTS
def data_dicts():
    sting_test = {"tap": ["tap.csv", "tap2.csv"], "background": "background.csv"}
    bicone_data = {
        0: ["TR001.csv", "TR007.csv", "TR008.csv"],
        2: ["TR002.csv"],
        4: ["TR003.csv"],
        6: ["TR004.csv", "TR005.csv", "TR006.csv"],
    }
    fin_config_data = {
        0: {
            0: ["TR009.csv", "TR030.csv"],
            5: ["TR035.csv"],
            10: ["TR012.csv", "TR026.csv", "TR027.csv"],
        },  # from 24 onwards its 40khz need to revised the cut off time
        5: {
            0: ["TR010.csv", "TR029.csv"],
            5: ["TR036.csv"],
            10: [
                "TR011.csv",
                "TR021.csv",
                "TR020.csv",
                "TR022.csv",
                "TR023.csv",
                "TR024.csv",
                "TR025.csv",
                "TR028.csv",
            ],
        },
    }
    flap_config_data = {
        0: {
            5: ["TR013.csv", "TR031.csv"],
            10: ["TR038.csv"],
            17.5: ["TR016.csv", "TR034.csv"],
        },
        5: {
            5: ["TR014.csv", "TR032.csv"],
            10: ["TR037.csv"],
            17.5: ["TR015.csv", "TR017.csv", "TR019.csv", "TR033.csv"],
        },
    }
    return sting_test, bicone_data, fin_config_data, flap_config_data


def tunnel_conditions():
    rho = 0.0381
    rho_std = 7.1 / 100 * rho  # in percent
    velocity = 1551.939
    velocity_std = 1.6 / 100 * velocity  # in percent
    diameter = 0.06
    diameter_std = np.sqrt(0.0005 / diameter) ** 2 * diameter
    # put all conditions in a dictionary
    condition_dict = {
        "rho": (rho, rho_std),
        "velocity": (velocity, velocity_std),
        "diameter": (diameter, diameter_std),
        "sting_diam": 15.875e-3,
        "mach": 8.11,
    }
    return condition_dict


def main():
    # NOTE: THIS MAIN FUNCTION OUTPUTS THE PLOTS FOR EXPERIMNTAL COMPARISON WITH NUMERICAL METHODS

    # data file dictionary
    # modify accordingly
    sting_test, bicone_data, fin_config_data, flap_config_data = data_dicts()
    condition_dict = tunnel_conditions()

    fin_post_df = process_data_to_dataframe(fin_config_data, condition_dict, "Fin")
    flap_post_df = process_data_to_dataframe(flap_config_data, condition_dict, "Flap")

    ## add deflection 0 values of fin to flaps
    flap_post_df = pd.concat(
        [flap_post_df, fin_post_df.query("deflection == 0")], ignore_index=True
    )
    # %% sort out data for inviscid code
    os.chdir("data")
    inviscid_fin = [
        pd.read_csv(f"{force.lower()}_invis_fin.csv", header=None)
        .set_index(pd.Index([0, 5, 10, 15, 20]))
        .rename(
            columns={
                0: -12.5,
                1: -10,
                2: -7.5,
                3: -5,
                4: -2.5,
                5: 0,
                6: 2.5,
                7: 5,
                8: 7.5,
                9: 10,
                10: 12.5,
            }
        )
        .reset_index()
        .melt(id_vars=["index"], var_name="deflection", value_name=force)
        for force in ["CL", "CD", "CM"]
    ]
    inviscid_fin = (
        inviscid_fin[0]
        .merge(inviscid_fin[1], on=["index", "deflection"])
        .merge(inviscid_fin[2], on=["index", "deflection"])
    )
    inviscid_fin.rename(columns={"index": "aoa"}, inplace=True)

    inviscid_flap = [
        pd.read_csv(f"{force.lower()}_invis_flap.csv", header=None)
        .set_index(pd.Index([0, 5, 10, 15, 20]))
        .rename(
            columns={
                0: -17.5,
                1: -15,
                2: -12.5,
                3: -10,
                4: -7.5,
                5: -5,
                6: 0,
                7: 5,
                8: 7.5,
                9: 10,
                10: 12.5,
                11: 15,
                12: 17.5,
            }
        )
        .reset_index()
        .melt(id_vars=["index"], var_name="deflection", value_name=force)
        for force in ["CL", "CD", "CM"]
    ]
    inviscid_flap = (
        inviscid_flap[0]
        .merge(inviscid_flap[1], on=["index", "deflection"])
        .merge(inviscid_flap[2], on=["index", "deflection"])
    )
    inviscid_flap.rename(columns={"index": "aoa"}, inplace=True)
    os.chdir("..")
    # %% sort out cfd data
    os.chdir("data")
    cfd_fin = [
        pd.read_excel(f"mach8_SST_{force}.xlsx", sheet_name="fin", index_col=0)
        .reset_index()
        .melt(id_vars=["index"], var_name="deflection", value_name=force)
        for force in ["CL", "CD", "CM"]
    ]
    # Merge the dataframes using the index and deflection columns
    cfd_fin = (
        cfd_fin[0]
        .merge(cfd_fin[1], on=["index", "deflection"])
        .merge(cfd_fin[2], on=["index", "deflection"])
    )
    cfd_fin.rename(columns={"index": "aoa"}, inplace=True)

    # now for flaps
    cfd_flap = [
        pd.read_excel(f"mach8_SST_{force}.xlsx", sheet_name="flap", index_col=0)
        .reset_index()
        .melt(id_vars=["index"], var_name="deflection", value_name=force)
        for force in ["CL", "CD", "CM"]
    ]
    cfd_flap = (cfd_flap[0].merge(cfd_flap[1], on=["index", "deflection"])).merge(
        cfd_flap[2], on=["index", "deflection"]
    )
    cfd_flap.rename(columns={"index": "aoa"}, inplace=True)

    os.chdir("..")
    # %% testing out plotting results
    # fin
    fin_plot = PostPlots(cfd_fin, fin_post_df, inviscid_fin)
    flap_plot = PostPlots(cfd_flap, flap_post_df, inviscid_flap)
    toggle = 1
    if toggle == 1:
        for aoa in [0, 5]:
            # flap
            flap_plot.plot_aoa(aoa, "lift", "flap", to_tikz=True)
            flap_plot.plot_aoa(aoa, "drag", "flap", to_tikz=True)
            flap_plot.plot_aoa(aoa, "moment", "flap", to_tikz=True)
    else:
        for aoa in [0, 5]:
            fin_plot.plot_aoa(
                aoa,
                "lift",
                "fin",to_tikz=True
            )
            fin_plot.plot_aoa(
                aoa,
                "drag",
                "fin",to_tikz=True
            )
            fin_plot.plot_aoa(
                aoa,
                "moment",
                "fin",to_tikz=True
            )

def peter_files():
    condition_dict = tunnel_conditions()
    peter_data_name = {
        5: {
            0: ["5701.xlsx","5704.xlsx","5705.xlsx"],
            5: ["5706.xlsx","5711.xlsx","5712.xlsx"],
            10: ["5713.xlsx","5717.xlsx","5718.xlsx"],
        },
    }

    # fin_a5_d0 = ExpPostProcess(peter_data_name[5][10][0], "Fin", 5, 10)
    # fin_a5_d0.plot_avg("axial",manual_cutoff=(0.0211,0.0460))
    peter_fin_post_df = process_data_to_dataframe(peter_data_name,condition_dict, "Fin", manual_cutoff=(0.0211,0.0460))
    sting_test, bicone_data, fin_config_data, flap_config_data = data_dicts()
    my_fin_post_df = process_data_to_dataframe(fin_config_data, condition_dict, "Fin")
    cfd_fin = [
        pd.read_excel(data_path/f"mach8_SST_{force}.xlsx", sheet_name="fin", index_col=0)
        .reset_index()
        .melt(id_vars=["index"], var_name="deflection", value_name=force)
        for force in ["CL", "CD", "CM"]
    ]
    # Merge the dataframes using the index and deflection columns
    cfd_fin = (
        cfd_fin[0]
        .merge(cfd_fin[1], on=["index", "deflection"])
        .merge(cfd_fin[2], on=["index", "deflection"])
    )
    cfd_fin.rename(columns={"index": "aoa"}, inplace=True)
    aoa_values = 5
    force_target = "CL"
    def plot_peter(aoa_values,force_target):
        force_label = {"CL":"Lift","CD":"Drag","CM":"Moment"}[force_target]
        max_val = {"CL": 0.38, "CD": 0.5, "CM": 0.2}[force_target]
    # Data sort
        cfd = cfd_fin.query(f"aoa == {str(aoa_values)}")
        exp = my_fin_post_df.query(f"aoa == {str(aoa_values)}")
        peter_exp = peter_fin_post_df.query(f"aoa == {str(aoa_values)}")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(
            cfd["deflection"],
            cfd[force_target],
            marker="^",
            linestyle="--",
            label=rf"CFD",
        )
        ax.plot(
            peter_exp["deflection"],
            peter_exp[force_target].apply(lambda x: x[0]),
            marker="v",
            c="r",
            linestyle="None",
            linewidth=1.5,
            label=f"Peter's Exp",
        )

        ax.errorbar(
            peter_exp["deflection"],
            peter_exp[force_target].apply(lambda x: x[0]),
            yerr=exp[force_target].apply(lambda x: x[1]),
            fmt="none",
            c="r",
            capsize=5,
        )
        ax.plot(
            exp["deflection"],
            exp[force_target].apply(lambda x: x[0]),
            marker="s",
            linestyle="None",
            c="k",
            label="Einar's Exp",
        )
        ax.errorbar(
            exp["deflection"],
            exp[force_target].apply(lambda x: x[0]),
            yerr=exp[force_target].apply(lambda x: x[1]),
            fmt="none",
            c="k",
            capsize=5,
        )

        ax.set_xlabel(bold_text("Deflection Angle (deg)"))
        ax.set_ylabel(bold_text(f"{force_label.capitalize()} Coefficient"))
        plt.ylim([0.0, max_val])
        plt.xlim([-0.1, cfd["deflection"].max() + 0.5])
        ax.legend(loc="best")
        fig.tight_layout()
        plt.show()
    [plot_peter(aoa_values,force_target) for force_target in ["CL","CD","CM"]]
    pass
def main_avg_timesieres():
    sting_test, bicone_data, fin_config_data, flap_config_data = data_dicts()
    fin_a5_d10 = fin_config_data[5][10][-1]
    fin_a0_d10 = fin_config_data[0][10][0]
    fin_a5_d10 = ExpPostProcess(fin_a5_d10, "Fin", 5, 10)
    fin_a0_d10 = ExpPostProcess(fin_a0_d10, "Fin", 0, 10)
    [fin_a5_d10.plot_avg(force) for force in ["normal", "axial", "moment"]]
    [fin_a0_d10.plot_avg(force) for force in ["normal", "axial", "moment"]]

    pass


def main_base_pres():
    # %% testing base pressure correction class
    mach = np.linspace(1, 8.4, 1000)
    base_pressure = BasePressureCalc(mach)
    honeywell = base_pressure.honeywell()
    gabeaud = base_pressure.gabeaud()
    love = base_pressure.love()
    fig, ax = plt.subplots(figsize=(6, 3))
    linew = 1.5

    ax.plot(mach, honeywell, label="Honeywell", linewidth=linew, c="k", linestyle=":")
    ax.plot(mach, gabeaud, label="Gabeaud", linewidth=linew, c="k", linestyle="--")
    ax.plot(mach, love, label="Love (Curve fit)", linewidth=linew, c="k", linestyle="-")
    ax.set_xlabel(bold_text(r"Freestream Mach Number, M\boldmath$_\infty$"))
    ax.set_ylabel(bold_text(r"Base Pressure Coefficient, C\boldmath$_{P_B}$"))
    # reverse the y axis
    ax.invert_yaxis()
    ax.legend()
    ax.set_xlim(1, 8.4)

    plt.tight_layout()

    plt.savefig("base_pressure.png", dpi=300)
    tikzplotlib.save(
        "base_pressure.tex",
        axis_width=".8\\textwidth",
        axis_height=".8*\\axisdefaultheight",
        table_row_sep=r"\\",
    )
    # plt.show(block=True)
    # %% testing aoa= 5, d = 10 for fin config
    pass


def main_freq():
    # %% frequency domain
    sting_test, bicone_data, fin_config_data, flap_config_data = data_dicts()
    force_to_check = "normal"
    tap = [
        ExpPostProcess(tap_i, config_i, "N/A", "N/A")
        for tap_i, config_i in zip(sting_test["tap"], ["Tap 1", "Tap 2"])
    ]
    background = ExpPostProcess(sting_test["background"], "Background", "N/A", "N/A")

    fin_0 = {
        keys: [
            ExpPostProcess(file, "Fin", 0, keys) for file in fin_config_data[0][keys]
        ]
        for keys in fin_config_data[0].keys()
    }

    flap_0 = {
        keys: [
            ExpPostProcess(file, "Flap", 0, keys) for file in flap_config_data[0][keys]
        ]
        for keys in flap_config_data[0].keys()
    }
    bicone_0 = [ExpPostProcess(file, "Bicone", 0, "N/A") for file in bicone_data[0]]

    def flatten(d):
        res = []  # Result list
        if isinstance(d, dict):
            for key, val in d.items():
                res.extend(flatten(val))
        elif isinstance(d, list):
            res = d
        else:
            raise TypeError("Undefined type for flatten: %s" % type(d))

        return res

    welch_res_0 = []
    tap_fft = [i.find_welch(force_to_check) for i in tap]
    background_fft = background.find_welch(force_to_check)
    [welch_res_0.append(i) for i in tap_fft]
    welch_res_0.append(background_fft)
    fin_0 = flatten(fin_0)
    flap_0 = flatten(flap_0)
    for i in fin_0:
        welch_res_0.append(i.find_welch(force_to_check))
    for i in flap_0:
        welch_res_0.append(i.find_welch(force_to_check))
    for i in bicone_0:
        welch_res_0.append(i.find_welch(force_to_check))

    welch_res_5 = []
    fin_5 = {
        keys: [
            ExpPostProcess(file, "Fin", 5, keys) for file in fin_config_data[5][keys]
        ]
        for keys in fin_config_data[5].keys()
    }
    flap = {
        keys: [
            ExpPostProcess(file, "Flap", 5, keys) for file in flap_config_data[5][keys]
        ]
        for keys in flap_config_data[5].keys()
    }
    welch_res_5.append(background_fft)
    [welch_res_5.append(i) for i in tap_fft]
    fin_5 = flatten(fin_5)
    flap_5 = flatten(flap)
    for i in fin_5:
        welch_res_5.append(i.find_welch(force_to_check))
    for i in flap_5:
        welch_res_5.append(i.find_welch(force_to_check))

    ExpPostProcess.plot_freq_domain(welch_res_0)

    ExpPostProcess.plot_freq_domain(welch_res_0, xmax=200, xmin=20)
    ExpPostProcess.plot_freq_domain(welch_res_5)
    ExpPostProcess.plot_freq_domain(welch_res_5, xmax=200, xmin=20)


def old_data():
    # old data
    data_dir = Path("data")
    # read file names from the directory with the extension .lvm
    files = [filename for filename in os.listdir(data_dir) if filename.endswith(".lvm")]
    option = files[3]
    print(option)
    test = ExpPostProcess_past(option, "past", "N/A", "N/A")
    test.plot_raw("axial")
    pass


def sharp_cone():
    rho = 0.0371
    rho_std = 7.1 / 100 * rho  # in percent
    velocity = 1553
    velocity_std = 1.6 / 100 * velocity  # in percent
    diameter = 17e-3
    diameter_std = np.sqrt(0.0005 / diameter) ** 2 * diameter
    # put all conditions in a dictionary
    condition_dict = {
        "rho": (rho, rho_std),
        "velocity": (velocity, velocity_std),
        "diameter": (diameter, diameter_std),
        "sting_diam": 12e-3,
        "mach": 8.2,
    }
    averaging_window = 500
    file_names = ["TR039.lvm", "TR040.lvm", "TR041.lvm", "TR042.lvm"]
    time_cut = (0.01907, 0.03094)
    test = ExpPostProcess(file_names[0], "Sharp Cone", 0.0, "N/A")
    test.averaging_window = averaging_window
    test.plot_avg("axial", manual_cutoff=time_cut)

    # force dictiory

    # List to store processed data
    data_list = []
    diameter_model = condition_dict["diameter"][0]
    diamter_sting = condition_dict["sting_diam"]
    ca_base_0 = BasePressureCalc(condition_dict["mach"]).gabeaud()
    post_obj = []
    force_coef_list = []
    for files in file_names:
        post_obj = ExpPostProcess(files, "Sharp Cone", 0.0, "N/A")
        post_obj.averaging_window = averaging_window
        # Process each file in the list for the current configuration
        coef = ErrorAnalysis(post_obj, condition_dict)
        force_coef_list.append(coef)

    force_avg, force_std_avg = ErrorAnalysis.repeat_avg(
        force_coef_list, "drag", time_cut
    )
    base_area = np.pi * diameter_model**2 / 4
    sting_area = np.pi * diamter_sting**2 / 4
    cl, cd = base_pressure_correction(
        force_avg, 0.0, ca_base_0, 0.0, base_area, sting_area
    )

    data_list.append(["CA", cd, force_std_avg])
    print(data_list)


if __name__ == "__main__":
    # main_avg_timesieres()
    # main_base_pres()
    # main()
    peter_files()
    # old_data()
    # main_base_pres()
    # sharp_cone()
    exit()
