"""
SOS-RAF membrane-translocation analysis for RAS-MAPK signaling.

Correspondence with the manuscript
----------------------------------
This script analyzes published single-cell measurements of epidermal growth
factor (EGF)-stimulated SOS and RAF membrane translocation under control
(DMSO) and MEK-inhibited (MEKi/Drug) conditions.

In the dual-fidelity framework:
    SOS is treated as the input variable X.
    RAF is treated as the output variable Z.
    INF is computed as the Gaussian mutual information between SOS and RAF.
    GMF is computed as the inverse 1D Gaussian 2-Wasserstein distance between
        the SOS and RAF marginal distributions.

Main calculations
-----------------
1. Load matched SOS and RAF single-cell trajectories for each condition.
2. Compute time-dependent mean, standard deviation, coefficient of variation,
   dynamic range, and SOS-RAF correlation coefficient.
3. Compute Gaussian MI, 2-WD, and inverse 2-WD at each time point.
4. Estimate uncertainty by paired bootstrap resampling of matched SOS-RAF
   single-cell pairs.
5. Export all figure data as tab-delimited .dat files for plotting in OriginPro.

Input files expected in the working directory
---------------------------------------------
DMSO/control:
    SOS_wt_DMSO_EGF10ng.csv
    RAF_wt_DMSO_EGF10ng.csv
MEKi/drug:
    SOS_wt_10nMMEKi_EGF10ng.csv
    RAF_wt_10nMMEKi_EGF10ng.csv

Output
------
The script exports processed figure data to:
    Exported-Data-final/

Author
------
Mintu Nandi
Universal Biology Institute, The University of Tokyo
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# USER SETTINGS
# ============================================================

FILE_MAP = {
    "DMSO": {
        "SOS": "SOS_wt_DMSO_EGF10ng.csv",
        "RAF": "RAF_wt_DMSO_EGF10ng.csv",
    },
    "Drug": {
        "SOS": "SOS_wt_10nMMEKi_EGF10ng.csv",
        "RAF": "RAF_wt_10nMMEKi_EGF10ng.csv",
    },
}

CONDITIONS = ["DMSO", "Drug"]

SPECIES_COLORS = {
    "SOS": "tab:blue",
    "RAF": "tab:orange",
}

CONDITION_COLORS = {
    "DMSO": "tab:green",
    "Drug": "tab:red",
}

REP_TIMES = [1, 5, 10, 30, 60]
REP_TIMES1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 28, 30, 40, 60]
HIST_BINS = 40

LINE_ALPHA = 0.12
LINE_WIDTH = 0.8

CV_EPS = 1e-12
INV_W2_EPS = 1e-12


N_BOOT = 500
BOOT_PERCENTILES = (16, 84)
BOOT_RANDOM_SEED = 12345


# ============================================================
# DATA LOADING
# ============================================================

def load_response_csv(path):
    """
    Load a CSV in the format:
      first column = time
      remaining columns = single-cell trajectories
    """
    df = pd.read_csv(path)

    first_col = df.columns[0]
    df = df.rename(columns={first_col: "time"})

    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")

    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])

    cell_cols = [c for c in df.columns if c != "time"]
    for c in cell_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("time").set_index("time")
    df.columns = [str(c) for c in df.columns]
    return df


def sorted_common_cells(cols1, cols2):
    common = list(set(cols1).intersection(set(cols2)))

    def keyfunc(x):
        try:
            return int(x)
        except ValueError:
            return x

    return sorted(common, key=keyfunc)


def load_all_data(file_map):
    """
    Returns:
        data[condition]["SOS"] = DataFrame(index=time, columns=common cell IDs)
        data[condition]["RAF"] = DataFrame(index=time, columns=common cell IDs)
    """
    data = {}

    for cond in CONDITIONS:
        sos = load_response_csv(file_map[cond]["SOS"])
        raf = load_response_csv(file_map[cond]["RAF"])

        common_times = sos.index.intersection(raf.index)
        common_times = np.sort(common_times.values.astype(float))

        common_cells = sorted_common_cells(sos.columns, raf.columns)

        sos = sos.loc[common_times, common_cells].copy()
        raf = raf.loc[common_times, common_cells].copy()

        data[cond] = {"SOS": sos, "RAF": raf}

    return data


# ============================================================
# BASIC HELPERS
# ============================================================

def nearest_time(index_like, target_time):
    idx = np.asarray(index_like, dtype=float)
    return idx[np.argmin(np.abs(idx - target_time))]


def get_values_at_time(data, condition, species, t):
    actual_t = nearest_time(data[condition][species].index, t)
    values = data[condition][species].loc[actual_t].values.astype(float)
    return actual_t, values


# ============================================================
# STATISTICS
# ============================================================

def compute_mean_std_cv_range(df):
    """
    For one species under one condition, compute across-cell statistics at each time:
      mean(t)
      std(t)
      CV(t) = std / |mean|
      dynamic_range(t) = max - min across cells
    """
    mean_ = df.mean(axis=1, skipna=True)
    std_ = df.std(axis=1, ddof=1, skipna=True)

    denom = mean_.abs()
    cv_ = std_ / denom
    cv_[denom < CV_EPS] = np.nan

    dynamic_range_ = df.max(axis=1, skipna=True) - df.min(axis=1, skipna=True)

    return {
        "mean": mean_,
        "std": std_,
        "cv": cv_,
        "dynamic_range": dynamic_range_,
    }


def compute_summary_stats(data):
    """
    stats[condition][species]["mean"/"std"/"cv"/"dynamic_range"]
    """
    stats = {}
    for cond in CONDITIONS:
        stats[cond] = {}
        for species in ["SOS", "RAF"]:
            stats[cond][species] = compute_mean_std_cv_range(data[cond][species])
    return stats


# ============================================================
# GAUSSIAN MI AND 2-WD
# ============================================================

def gaussian_mi_bits(x, z):
    """
    Gaussian MI in bits:
        I = -1/2 * log2(1 - rho^2)
    """
    x = np.asarray(x)
    z = np.asarray(z)

    mask = np.isfinite(x) & np.isfinite(z)
    x = x[mask]
    z = z[mask]

    if len(x) < 3:
        return np.nan

    sx = np.std(x, ddof=1)
    sz = np.std(z, ddof=1)

    if sx < CV_EPS or sz < CV_EPS:
        return 0.0

    rho = np.corrcoef(x, z)[0, 1]
    rho = np.clip(rho, -0.999999999999, 0.999999999999)

    return -0.5 * np.log2(1.0 - rho**2)


def gaussian_w2_1d(x, z):
    """
    1D Gaussian 2-Wasserstein distance between marginals:
        W2^2 = (mu_x - mu_z)^2 + (sigma_x - sigma_z)^2
    """
    x = np.asarray(x)
    z = np.asarray(z)

    x = x[np.isfinite(x)]
    z = z[np.isfinite(z)]

    if len(x) < 2 or len(z) < 2:
        return np.nan

    mu_x = np.mean(x)
    mu_z = np.mean(z)
    sigma_x = np.std(x, ddof=1)
    sigma_z = np.std(z, ddof=1)

    w2_sq = (mu_x - mu_z)**2 + (sigma_x - sigma_z)**2
    return np.sqrt(max(w2_sq, 0.0))


def gaussian_inv_w2_1d(x, z, eps=INV_W2_EPS):
    w2 = gaussian_w2_1d(x, z)
    if not np.isfinite(w2):
        return np.nan
    return 1.0 / (w2 + eps)



def gaussian_rho(x, z):
    """
    Pearson correlation coefficient between SOS and RAF across cells.
    Uses the same matched samples as the Gaussian MI calculation.
    """
    x = np.asarray(x)
    z = np.asarray(z)

    mask = np.isfinite(x) & np.isfinite(z)
    x = x[mask]
    z = z[mask]

    if len(x) < 3:
        return np.nan

    sx = np.std(x, ddof=1)
    sz = np.std(z, ddof=1)

    if sx < CV_EPS or sz < CV_EPS:
        return 0.0

    rho = np.corrcoef(x, z)[0, 1]
    return np.clip(rho, -1.0, 1.0)


def bootstrap_metric_error(values, point_estimate, nonnegative=True,
                           percentiles=BOOT_PERCENTILES):
    """
    Convert a bootstrap distribution into a single symmetric error bar.

    The error is chosen conservatively from the percentile interval:
        err = min(point_estimate - q_low, q_high - point_estimate)

    If nonnegative=True, the lower percentile is clipped at 0 and the final
    error is also clipped so that:
        err <= point_estimate

    Therefore the lower error bar never goes below zero.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0 or not np.isfinite(point_estimate):
        return np.nan

    q_low, q_high = np.percentile(values, percentiles)

    if nonnegative:
        q_low = max(0.0, q_low)

    err_low = max(0.0, point_estimate - q_low)
    err_high = max(0.0, q_high - point_estimate)

    err = min(err_low, err_high)

    if nonnegative:
        err = min(err, point_estimate)

    return err


def bootstrap_mi_and_invw2(x, z, n_boot=N_BOOT, rng=None):
    """
    Paired bootstrap for Gaussian MI and inverse W2.

    Bootstrap resamples matched SOS-RAF cell pairs with replacement.
    Returns single symmetric bootstrap errors.
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)

    mask = np.isfinite(x) & np.isfinite(z)
    x = x[mask]
    z = z[mask]

    n = len(x)
    if n < 3:
        return {
            "MI": np.nan,
            "MI_err": np.nan,
            "W2": np.nan,
            "invW2": np.nan,
            "invW2_err": np.nan,
        }

    if rng is None:
        rng = np.random.default_rng()

    mi_point = gaussian_mi_bits(x, z)
    w2_point = gaussian_w2_1d(x, z)
    invw2_point = gaussian_inv_w2_1d(x, z)

    mi_boot = []
    invw2_boot = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        zb = z[idx]

        mi_b = gaussian_mi_bits(xb, zb)
        invw2_b = gaussian_inv_w2_1d(xb, zb)

        if np.isfinite(mi_b):
            mi_boot.append(mi_b)
        if np.isfinite(invw2_b):
            invw2_boot.append(invw2_b)

    mi_err = bootstrap_metric_error(
        mi_boot, mi_point, nonnegative=True
    )
    invw2_err = bootstrap_metric_error(
        invw2_boot, invw2_point, nonnegative=True
    )

    return {
        "MI": mi_point,
        "MI_err": mi_err,
        "W2": w2_point,
        "invW2": invw2_point,
        "invW2_err": invw2_err,
    }

def compute_explanatory_metrics(data):
    """
    Returns only:
        rho
        mean_SOS
        mean_RAF
        std_SOS
        std_RAF
    """
    times = data[CONDITIONS[0]]["SOS"].index

    rho_df = pd.DataFrame(index=times, columns=CONDITIONS, dtype=float)
    mean_sos_df = pd.DataFrame(index=times, columns=CONDITIONS, dtype=float)
    mean_raf_df = pd.DataFrame(index=times, columns=CONDITIONS, dtype=float)
    std_sos_df = pd.DataFrame(index=times, columns=CONDITIONS, dtype=float)
    std_raf_df = pd.DataFrame(index=times, columns=CONDITIONS, dtype=float)

    for cond in CONDITIONS:
        sos_df = data[cond]["SOS"]
        raf_df = data[cond]["RAF"]

        for t in times:
            x = sos_df.loc[t].values.astype(float)
            z = raf_df.loc[t].values.astype(float)

            x_valid = x[np.isfinite(x)]
            z_valid = z[np.isfinite(z)]

            rho_df.loc[t, cond] = gaussian_rho(x, z)
            mean_sos_df.loc[t, cond] = np.mean(x_valid) if len(x_valid) > 0 else np.nan
            mean_raf_df.loc[t, cond] = np.mean(z_valid) if len(z_valid) > 0 else np.nan
            std_sos_df.loc[t, cond] = np.std(x_valid, ddof=1) if len(x_valid) > 1 else np.nan
            std_raf_df.loc[t, cond] = np.std(z_valid, ddof=1) if len(z_valid) > 1 else np.nan

    return {
        "rho": rho_df,
        "mean_SOS": mean_sos_df,
        "mean_RAF": mean_raf_df,
        "std_SOS": std_sos_df,
        "std_RAF": std_raf_df,
    }


def compute_time_metrics(data):
    """
    Returns bootstrap-based point estimates and single symmetric error bars.

    metrics contains:
        MI
        MI_err
        W2
        invW2
        invW2_err
    """
    times = data[CONDITIONS[0]]["SOS"].index

    mi_df = pd.DataFrame(index=times, columns=CONDITIONS, dtype=float)
    mi_err_df = pd.DataFrame(index=times, columns=CONDITIONS, dtype=float)

    w2_df = pd.DataFrame(index=times, columns=CONDITIONS, dtype=float)

    invw2_df = pd.DataFrame(index=times, columns=CONDITIONS, dtype=float)
    invw2_err_df = pd.DataFrame(index=times, columns=CONDITIONS, dtype=float)

    rng = np.random.default_rng(BOOT_RANDOM_SEED)

    for cond in CONDITIONS:
        sos = data[cond]["SOS"]
        raf = data[cond]["RAF"]

        for t in times:
            x = sos.loc[t].values.astype(float)
            z = raf.loc[t].values.astype(float)

            out = bootstrap_mi_and_invw2(x, z, n_boot=N_BOOT, rng=rng)

            mi_df.loc[t, cond] = out["MI"]
            mi_err_df.loc[t, cond] = out["MI_err"]

            w2_df.loc[t, cond] = out["W2"]

            invw2_df.loc[t, cond] = out["invW2"]
            invw2_err_df.loc[t, cond] = out["invW2_err"]

    return {
        "MI": mi_df,
        "MI_err": mi_err_df,
        "W2": w2_df,
        "invW2": invw2_df,
        "invW2_err": invw2_err_df,
    }


# ============================================================
# PIPELINE
# ============================================================

def build_all_tables(data):
    stats = compute_summary_stats(data)
    metrics = compute_time_metrics(data)
    explain = compute_explanatory_metrics(data)
    return stats, metrics, explain


# ============================================================
# AX-LEVEL DRAWING HELPERS
# ============================================================

def draw_time_series_on_ax(ax, data, condition, title):
    for species in ["SOS", "RAF"]:
        df = data[condition][species]
        time = df.index.values.astype(float)

        for col in df.columns:
            ax.plot(
                time,
                df[col].values,
                color=SPECIES_COLORS[species],
                alpha=LINE_ALPHA,
                lw=LINE_WIDTH
            )

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Response")
    ax.set_title(title)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=SPECIES_COLORS["SOS"], lw=2, label="SOS"),
        Line2D([0], [0], color=SPECIES_COLORS["RAF"], lw=2, label="RAF"),
    ]
    ax.legend(handles=handles, frameon=False)


def draw_marginal_hist_on_ax(ax, data, condition, species, rep_times, title, bins=HIST_BINS):
    cmap = plt.cm.viridis
    colors = [cmap(i) for i in np.linspace(0.15, 0.85, len(rep_times))]

    for c, t in zip(colors, rep_times):
        actual_t, vals = get_values_at_time(data, condition, species, t)
        vals = vals[np.isfinite(vals)]

        ax.hist(
            vals,
            bins=bins,
            alpha=0.35,
            color=c,
            edgecolor="black",
            linewidth=0.4,
            label=f"t = {actual_t:g} min"
        )

    ax.set_xlabel(f"{species} response")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(frameon=False)


def draw_dynamic_range_on_ax(ax, stats, condition, title):
    for species in ["SOS", "RAF"]:
        dr = stats[condition][species]["dynamic_range"]
        ax.plot(
            dr.index,
            dr.values,
            color=SPECIES_COLORS[species],
            lw=2,
            label=species
        )

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Dynamic range across cells (max - min)")
    ax.set_title(title)
    ax.legend(frameon=False)


def draw_cv_on_ax(ax, stats, condition, title):
    for species in ["SOS", "RAF"]:
        cv = stats[condition][species]["cv"]
        ax.plot(
            cv.index,
            cv.values,
            color=SPECIES_COLORS[species],
            lw=2,
            label=species
        )

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("CV = std / |mean|")
    ax.set_title(title)
    ax.legend(frameon=False)


def draw_metric_on_ax(ax, metrics, metric_name, title, ylabel):
    df = metrics[metric_name]
    err_df = metrics[f"{metric_name}_err"]

    for cond in CONDITIONS:
        x = df.index.values.astype(float)
        y = df[cond].values.astype(float)
        yerr = err_df[cond].values.astype(float)

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            color=CONDITION_COLORS[cond],
            lw=1.8,
            elinewidth=1.0,
            capsize=2,
            fmt='-',
            label=cond
        )

    ax.set_xlabel("Time (min)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)


def draw_joint_distribution_on_ax(ax, data, condition, t, title):
    actual_t_sos, x = get_values_at_time(data, condition, "SOS", t)
    actual_t_raf, y = get_values_at_time(data, condition, "RAF", t)

    mask = np.isfinite(x) & np.isfinite(y)
    ax.scatter(
        x[mask],
        y[mask],
        s=16,
        alpha=0.45,
        color=CONDITION_COLORS[condition],
        edgecolors="none"
    )

    ax.set_xlabel("SOS")
    ax.set_ylabel("RAF")
    ax.set_title(title)
    return actual_t_sos


def draw_mi_vs_invw2_on_ax(ax, metrics, t, title):
    actual_t = nearest_time(metrics["MI"].index, t)

    for cond in CONDITIONS:
        x = metrics["invW2"].loc[actual_t, cond]
        y = metrics["MI"].loc[actual_t, cond]

        xerr = metrics["invW2_err"].loc[actual_t, cond]
        yerr = metrics["MI_err"].loc[actual_t, cond]

        ax.errorbar(
            x, y,
            xerr=xerr,
            yerr=yerr,
            fmt='o',
            ms=6,
            color=CONDITION_COLORS[cond],
            elinewidth=1.0,
            capsize=2,
            label=cond
        )

        ax.annotate(cond, (x, y), xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel(r"$W_D^{-1}(\mathrm{SOS}, \mathrm{RAF})$")
    ax.set_ylabel("MI(SOS;RAF) [bits]")
    ax.set_title(title)
    ax.legend(frameon=False)


# ============================================================
# COMBINED FIGURE FUNCTIONS
# ============================================================

def plot_figure1(data):
    """
    Figure 1 with panels:
      (a) DMSO
      (b) Drug
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

    draw_time_series_on_ax(axes[0], data, "DMSO", "(a) DMSO")
    draw_time_series_on_ax(axes[1], data, "Drug", "(b) MEKi")

    fig.suptitle("Figure 1: Single-cell time series", y=1.02)
    fig.tight_layout()
    return fig, axes


def plot_figure2(data, rep_times=REP_TIMES):
    """
    Figure 2 with panels:
      (a) SOS DMSO
      (b) RAF DMSO
      (c) SOS Drug
      (d) RAF Drug
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    draw_marginal_hist_on_ax(axes[0, 0], data, "DMSO", "SOS", rep_times, "(a) SOS, DMSO")
    draw_marginal_hist_on_ax(axes[0, 1], data, "DMSO", "RAF", rep_times, "(b) RAF, DMSO")
    draw_marginal_hist_on_ax(axes[1, 0], data, "Drug", "SOS", rep_times, "(c) SOS, MEKi")
    draw_marginal_hist_on_ax(axes[1, 1], data, "Drug", "RAF", rep_times, "(d) RAF, MEKi")

    fig.suptitle("Figure 2: Marginal distributions", y=1.02)
    fig.tight_layout()
    return fig, axes


def plot_figure3(data, rep_times=REP_TIMES):
    """
    Figure 3:
      rows = conditions
      columns = times
    """
    ncols = len(rep_times)
    fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 8), sharex=True, sharey=True)

    # determine global limits
    x_all, y_all = [], []
    for cond in CONDITIONS:
        for t in rep_times:
            _, x = get_values_at_time(data, cond, "SOS", t)
            _, y = get_values_at_time(data, cond, "RAF", t)
            mask = np.isfinite(x) & np.isfinite(y)
            x_all.append(x[mask])
            y_all.append(y[mask])

    x_all = np.concatenate(x_all)
    y_all = np.concatenate(y_all)

    xpad = 0.05 * (np.nanmax(x_all) - np.nanmin(x_all) + 1e-12)
    ypad = 0.05 * (np.nanmax(y_all) - np.nanmin(y_all) + 1e-12)

    xlim = (np.nanmin(x_all) - xpad, np.nanmax(x_all) + xpad)
    ylim = (np.nanmin(y_all) - ypad, np.nanmax(y_all) + ypad)

    for i, cond in enumerate(CONDITIONS):
        for j, t in enumerate(rep_times):
            actual_t = draw_joint_distribution_on_ax(
                axes[i, j],
                data,
                cond,
                t,
                title=f"{cond}, t={nearest_time(data[cond]['SOS'].index, t):g} min"
            )
            axes[i, j].set_xlim(*xlim)
            axes[i, j].set_ylim(*ylim)

    fig.suptitle("Figure 3: Joint distributions of SOS and RAF", y=1.02)
    fig.tight_layout()
    return fig, axes


def plot_figure4(stats):
    """
    Figure 4 with panels:
      (a) dynamic range of SOS: DMSO vs Drug
      (b) dynamic range of RAF: DMSO vs Drug
      (c) CV of SOS: DMSO vs Drug
      (d) CV of RAF: DMSO vs Drug
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

    # (a) dynamic range of SOS
    ax = axes[0, 0]
    for cond in CONDITIONS:
        dr = stats[cond]["SOS"]["dynamic_range"]
        ax.plot(
            dr.index,
            dr.values,
            color=CONDITION_COLORS[cond],
            lw=2,
            label=cond
        )
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Dynamic range across cells (max - min)")
    ax.set_title("(a) Dynamic range of SOS")
    ax.legend(frameon=False)

    # (b) dynamic range of RAF
    ax = axes[0, 1]
    for cond in CONDITIONS:
        dr = stats[cond]["RAF"]["dynamic_range"]
        ax.plot(
            dr.index,
            dr.values,
            color=CONDITION_COLORS[cond],
            lw=2,
            label=cond
        )
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Dynamic range across cells (max - min)")
    ax.set_title("(b) Dynamic range of RAF")
    ax.legend(frameon=False)

    # (c) CV of SOS
    ax = axes[1, 0]
    for cond in CONDITIONS:
        cv = stats[cond]["SOS"]["cv"]
        ax.plot(
            cv.index,
            cv.values,
            color=CONDITION_COLORS[cond],
            lw=2,
            label=cond
        )
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("CV = std / |mean|")
    ax.set_title("(c) CV of SOS")
    ax.legend(frameon=False)

    # (d) CV of RAF
    ax = axes[1, 1]
    for cond in CONDITIONS:
        cv = stats[cond]["RAF"]["cv"]
        ax.plot(
            cv.index,
            cv.values,
            color=CONDITION_COLORS[cond],
            lw=2,
            label=cond
        )
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("CV = std / |mean|")
    ax.set_title("(d) CV of RAF")
    ax.legend(frameon=False)

    fig.suptitle("Figure 4: Dynamic range and CV comparison between DMSO and MEKi", y=1.02)
    fig.tight_layout()
    return fig, axes


def plot_figure5(metrics):
    """
    Figure 5 with panels:
      (a) MI
      (b) inverse WD
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    draw_metric_on_ax(
        axes[0],
        metrics,
        metric_name="MI",
        title="(a) Mutual information",
        ylabel="MI(SOS;RAF) [bits]"
    )

    draw_metric_on_ax(
        axes[1],
        metrics,
        metric_name="invW2",
        title=r"(b) $W_D^{-1}$",
        ylabel=r"$W_D^{-1}(\mathrm{SOS}, \mathrm{RAF})$"
    )

    fig.suptitle(r"Figure 5: Temporal variation of MI and $W_D^{-1}$", y=1.02)
    fig.tight_layout()
    return fig, axes


def plot_figure6(metrics, rep_times=REP_TIMES):
    """
    Figure 6:
      MI vs inverse W2 at selected times
    """
    n = len(rep_times)
    nrows = int(np.ceil(n / 4))
    ncols = 6

    fig, axes = plt.subplots(nrows, ncols, figsize=(30, 4.5 * nrows))
    axes = np.array(axes).reshape(-1)

    # common axis limits
    x_all, y_all = [], []
    for t in rep_times:
        actual_t = nearest_time(metrics["MI"].index, t)
        for cond in CONDITIONS:
            x = metrics["invW2"].loc[actual_t, cond]
            y = metrics["MI"].loc[actual_t, cond]
            if np.isfinite(x) and np.isfinite(y):
                x_all.append(x)
                y_all.append(y)

    if len(x_all) > 0:
        x_all = np.array(x_all)
        y_all = np.array(y_all)
        xpad = 0.1 * (x_all.max() - x_all.min() + 1e-12)
        ypad = 0.1 * (y_all.max() - y_all.min() + 1e-12)
        xlim = (x_all.min() - xpad, x_all.max() + xpad)
        ylim = (y_all.min() - ypad, y_all.max() + ypad)
    else:
        xlim = None
        ylim = None

    for ax, t in zip(axes, rep_times):
        actual_t = nearest_time(metrics["MI"].index, t)
        draw_mi_vs_invw2_on_ax(ax, metrics, t, title=f"t = {actual_t:g} min")
        if xlim is not None:
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

    for k in range(len(rep_times), len(axes)):
        axes[k].axis("off")

    fig.suptitle("Figure 6: MI vs inverse 2-WD", y=1.02)
    fig.tight_layout()
    return fig, axes

def plot_figure7(explain):
    """
    Figure 7:
      (a) rho for DMSO and Drug
      (b) mean SOS and RAF for DMSO
      (c) mean SOS and RAF for Drug
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

    # (a) rho
    ax = axes[0]
    for cond in CONDITIONS:
        ax.plot(
            explain["rho"].index,
            explain["rho"][cond].values,
            color=CONDITION_COLORS[cond],
            lw=2,
            label=cond
        )
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Correlation coefficient")
    ax.set_title("(a) SOS-RAF correlation")
    ax.legend(frameon=False)

    # (b) means for DMSO
    ax = axes[1]
    ax.plot(
        explain["mean_SOS"].index,
        explain["mean_SOS"]["DMSO"].values,
        color=SPECIES_COLORS["SOS"],
        lw=2,
        label="SOS"
    )
    ax.plot(
        explain["mean_RAF"].index,
        explain["mean_RAF"]["DMSO"].values,
        color=SPECIES_COLORS["RAF"],
        lw=2,
        label="RAF"
    )
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Mean response")
    ax.set_title("(b) Means, DMSO")
    ax.legend(frameon=False)

    # (c) means for Drug
    ax = axes[2]
    ax.plot(
        explain["mean_SOS"].index,
        explain["mean_SOS"]["Drug"].values,
        color=SPECIES_COLORS["SOS"],
        lw=2,
        label="SOS"
    )
    ax.plot(
        explain["mean_RAF"].index,
        explain["mean_RAF"]["Drug"].values,
        color=SPECIES_COLORS["RAF"],
        lw=2,
        label="RAF"
    )
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Mean response")
    ax.set_title("(c) Means, MEKi")
    ax.legend(frameon=False)

    fig.suptitle("Figure 7: Correlation and means", y=1.02)
    fig.tight_layout()
    return fig, axes

def plot_figure8(explain):
    """
    Figure 8:
      (a) std SOS and RAF for DMSO
      (b) std SOS and RAF for Drug
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    # (a) stds for DMSO
    ax = axes[0]
    ax.plot(
        explain["std_SOS"].index,
        explain["std_SOS"]["DMSO"].values,
        color=SPECIES_COLORS["SOS"],
        lw=2,
        label="SOS"
    )
    ax.plot(
        explain["std_RAF"].index,
        explain["std_RAF"]["DMSO"].values,
        color=SPECIES_COLORS["RAF"],
        lw=2,
        label="RAF"
    )
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Standard deviation")
    ax.set_title("(a) Standard deviations, DMSO")
    ax.legend(frameon=False)

    # (b) stds for Drug
    ax = axes[1]
    ax.plot(
        explain["std_SOS"].index,
        explain["std_SOS"]["Drug"].values,
        color=SPECIES_COLORS["SOS"],
        lw=2,
        label="SOS"
    )
    ax.plot(
        explain["std_RAF"].index,
        explain["std_RAF"]["Drug"].values,
        color=SPECIES_COLORS["RAF"],
        lw=2,
        label="RAF"
    )
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Standard deviation")
    ax.set_title("(b) Standard deviations, MEKi")
    ax.legend(frameon=False)

    fig.suptitle("Figure 8: Standard deviations", y=1.02)
    fig.tight_layout()
    return fig, axes

# ============================================================
# EXPORT TO .DAT FILES
# ============================================================

def write_dat(df, filepath):
    """
    Write a tab-delimited .dat file compatible with Origin Pro.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        filepath,
        sep="\t",
        index=False,
        na_rep="NaN",
        float_format="%.10g"
    )


def sanitize_time_label(t):
    """
    Convert time value into a safe filename token.
    Examples:
        1   -> '1'
        2.0 -> '2'
        2.5 -> '2p5'
        -1  -> 'm1'
    """
    t = float(t)
    if t.is_integer():
        s = str(int(t))
    else:
        s = str(t).replace(".", "p")
    s = s.replace("-", "m")
    return s


# ------------------------------------------------------------
# PDF BUILDERS
# ------------------------------------------------------------

def build_1d_pdf_table(values, bins=HIST_BINS):
    """
    Build 1D normalized PDF table for histogram plotting in Origin.
    Output columns:
        Response, Normalized_PDF
    Here Response is the bin center.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return pd.DataFrame(columns=["Response", "Normalized_PDF"])

    counts, edges = np.histogram(values, bins=bins, density=False)
    pdf = counts / counts.sum()
    centers = 0.5 * (edges[:-1] + edges[1:])

    return pd.DataFrame({
        "Response": centers,
        "Normalized_PDF": pdf
    })


def build_2d_pdf_table(x, y, bins=HIST_BINS):
    """
    Build 2D normalized PDF table for joint-distribution plotting in Origin.
    Output columns:
        SOS_Response, RAF_Response, Normalized_PDF

    The SOS_Response and RAF_Response values are bin centers.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return pd.DataFrame(columns=["SOS_Response", "RAF_Response", "Normalized_PDF"])

    counts, xedges, yedges = np.histogram2d(x, y, bins=bins, density=False)

    pdf = counts / counts.sum()

    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])

    Xc, Yc = np.meshgrid(xcenters, ycenters, indexing="ij")

    return pd.DataFrame({
        "SOS_Response": Xc.ravel(),
        "RAF_Response": Yc.ravel(),
        "Normalized_PDF": pdf.ravel()
    })


# ------------------------------------------------------------
# Figure 1 exports
# ------------------------------------------------------------

def export_fig1_dat(data, outdir="origin_dat"):
    """
    Export 4 files:
      Fig1_DMSO_SOS_timeseries.dat
      Fig1_DMSO_RAF_timeseries.dat
      Fig1_Drug_SOS_timeseries.dat
      Fig1_Drug_RAF_timeseries.dat

    Wide format:
      first column = Time_min
      remaining columns = Cell_...
    """
    outdir = Path(outdir)
    exported = {}

    for cond in CONDITIONS:
        for species in ["SOS", "RAF"]:
            df = data[cond][species].copy()
            out = df.copy()
            out.insert(0, "Time_min", out.index.values.astype(float))
            out = out.reset_index(drop=True)

            filepath = outdir / f"Fig1_{cond}_{species}_timeseries.dat"
            write_dat(out, filepath)
            exported[f"{cond}_{species}"] = out

    return exported


# ------------------------------------------------------------
# Figure 2 exports
# ------------------------------------------------------------

def export_fig2_dat(data, rep_times, outdir="origin_dat", bins=HIST_BINS):
    """
    Export one file per:
      condition × species × time

    Example:
      Fig2_DMSO_SOS_t1.dat
      Fig2_Drug_RAF_t30.dat

    Format:
      Response    Normalized_PDF
    """
    outdir = Path(outdir)
    exported = {}

    for cond in CONDITIONS:
        for species in ["SOS", "RAF"]:
            df = data[cond][species]

            for t in rep_times:
                vals = df.loc[t].values.astype(float)
                out = build_1d_pdf_table(vals, bins=bins)

                t_label = sanitize_time_label(t)
                filepath = outdir / f"Fig2_{cond}_{species}_t{t_label}.dat"
                write_dat(out, filepath)

                exported[f"{cond}_{species}_t{t_label}"] = out

    return exported


# ------------------------------------------------------------
# Figure 3 exports
# ------------------------------------------------------------

def export_fig3_dat(data, rep_times, outdir="origin_dat", bins=HIST_BINS):
    """
    Export one file per:
      condition × time

    Example:
      Fig3_DMSO_t1_joint.dat
      Fig3_Drug_t30_joint.dat

    Format:
      SOS_Response    RAF_Response    Normalized_PDF
    """
    outdir = Path(outdir)
    exported = {}

    for cond in CONDITIONS:
        sos_df = data[cond]["SOS"]
        raf_df = data[cond]["RAF"]

        for t in rep_times:
            sos_vals = sos_df.loc[t].values.astype(float)
            raf_vals = raf_df.loc[t].values.astype(float)

            out = build_2d_pdf_table(sos_vals, raf_vals, bins=bins)

            t_label = sanitize_time_label(t)
            filepath = outdir / f"Fig3_{cond}_t{t_label}_joint.dat"
            write_dat(out, filepath)

            exported[f"{cond}_t{t_label}"] = out

    return exported


# ------------------------------------------------------------
# Figure 4 exports
# ------------------------------------------------------------

def export_fig4_dat(stats, outdir="origin_dat"):
    """
    Export one file per panel:
      Fig4a_SOS_dynamic_range.dat
      Fig4b_RAF_dynamic_range.dat
      Fig4c_SOS_CV.dat
      Fig4d_RAF_CV.dat
    """
    outdir = Path(outdir)
    exported = {}

    times = stats["DMSO"]["SOS"]["dynamic_range"].index.values.astype(float)

    out_4a = pd.DataFrame({
        "Time_min": times,
        "DMSO": stats["DMSO"]["SOS"]["dynamic_range"].values,
        "Drug": stats["Drug"]["SOS"]["dynamic_range"].values,
    })
    write_dat(out_4a, outdir / "Fig4a_SOS_dynamic_range.dat")
    exported["Fig4a"] = out_4a

    out_4b = pd.DataFrame({
        "Time_min": times,
        "DMSO": stats["DMSO"]["RAF"]["dynamic_range"].values,
        "Drug": stats["Drug"]["RAF"]["dynamic_range"].values,
    })
    write_dat(out_4b, outdir / "Fig4b_RAF_dynamic_range.dat")
    exported["Fig4b"] = out_4b

    out_4c = pd.DataFrame({
        "Time_min": times,
        "DMSO": stats["DMSO"]["SOS"]["cv"].values,
        "Drug": stats["Drug"]["SOS"]["cv"].values,
    })
    write_dat(out_4c, outdir / "Fig4c_SOS_CV.dat")
    exported["Fig4c"] = out_4c

    out_4d = pd.DataFrame({
        "Time_min": times,
        "DMSO": stats["DMSO"]["RAF"]["cv"].values,
        "Drug": stats["Drug"]["RAF"]["cv"].values,
    })
    write_dat(out_4d, outdir / "Fig4d_RAF_CV.dat")
    exported["Fig4d"] = out_4d

    return exported


# ------------------------------------------------------------
# Figure 5 exports
# ------------------------------------------------------------

def export_fig5_dat(metrics, outdir="origin_dat"):
    """
    Export one file per panel:
      Fig5a_MI_vs_time.dat
      Fig5b_WDinv_vs_time.dat
    """
    outdir = Path(outdir)
    exported = {}

    times = metrics["MI"].index.values.astype(float)

    out_5a = pd.DataFrame({
        "Time_min": times,
        "DMSO": metrics["MI"]["DMSO"].values,
        "DMSO_err": metrics["MI_err"]["DMSO"].values,
        "Drug": metrics["MI"]["Drug"].values,
        "Drug_err": metrics["MI_err"]["Drug"].values,
    })
    write_dat(out_5a, outdir / "Fig5a_MI_vs_time.dat")
    exported["Fig5a"] = out_5a

    out_5b = pd.DataFrame({
        "Time_min": times,
        "DMSO": metrics["invW2"]["DMSO"].values,
        "DMSO_err": metrics["invW2_err"]["DMSO"].values,
        "Drug": metrics["invW2"]["Drug"].values,
        "Drug_err": metrics["invW2_err"]["Drug"].values,
    })
    write_dat(out_5b, outdir / "Fig5b_WDinv_vs_time.dat")
    exported["Fig5b"] = out_5b

    return exported


# ------------------------------------------------------------
# Figure 6 exports
# ------------------------------------------------------------

def export_fig6_dat(metrics, rep_times, outdir="origin_dat"):
    """
    Export one file per time.

    Format:
      Condition    WDinv    WDinv_err    MI    MI_err
    """
    outdir = Path(outdir)
    exported = {}

    for t in rep_times:
        out = pd.DataFrame({
            "Condition": ["DMSO", "Drug"],
            "WDinv": [
                metrics["invW2"].loc[t, "DMSO"],
                metrics["invW2"].loc[t, "Drug"],
            ],
            "WDinv_err": [
                metrics["invW2_err"].loc[t, "DMSO"],
                metrics["invW2_err"].loc[t, "Drug"],
            ],
            "MI": [
                metrics["MI"].loc[t, "DMSO"],
                metrics["MI"].loc[t, "Drug"],
            ],
            "MI_err": [
                metrics["MI_err"].loc[t, "DMSO"],
                metrics["MI_err"].loc[t, "Drug"],
            ],
        })

        t_label = sanitize_time_label(t)
        filepath = outdir / f"Fig6_t{t_label}_dual_fidelity.dat"
        write_dat(out, filepath)

        exported[f"t{t_label}"] = out

    return exported

# ------------------------------------------------------------
# Figure 7 exports
# ------------------------------------------------------------

def export_fig7_dat(explain, outdir="origin_dat"):
    """
    Export:
      Fig7a_rho.dat
      Fig7b_mean_DMSO.dat
      Fig7c_mean_Drug.dat
    """
    outdir = Path(outdir)
    exported = {}

    times = explain["rho"].index.values.astype(float)

    out_7a = pd.DataFrame({
        "Time_min": times,
        "DMSO": explain["rho"]["DMSO"].values,
        "Drug": explain["rho"]["Drug"].values,
    })
    write_dat(out_7a, outdir / "Fig7a_rho.dat")
    exported["Fig7a"] = out_7a

    out_7b = pd.DataFrame({
        "Time_min": times,
        "SOS": explain["mean_SOS"]["DMSO"].values,
        "RAF": explain["mean_RAF"]["DMSO"].values,
    })
    write_dat(out_7b, outdir / "Fig7b_mean_DMSO.dat")
    exported["Fig7b"] = out_7b

    out_7c = pd.DataFrame({
        "Time_min": times,
        "SOS": explain["mean_SOS"]["Drug"].values,
        "RAF": explain["mean_RAF"]["Drug"].values,
    })
    write_dat(out_7c, outdir / "Fig7c_mean_Drug.dat")
    exported["Fig7c"] = out_7c

    return exported

# ------------------------------------------------------------
# Figure 8 exports
# ------------------------------------------------------------

def export_fig8_dat(explain, outdir="origin_dat"):
    """
    Export:
      Fig8a_std_DMSO.dat
      Fig8b_std_Drug.dat
    """
    outdir = Path(outdir)
    exported = {}

    times = explain["rho"].index.values.astype(float)

    out_8a = pd.DataFrame({
        "Time_min": times,
        "SOS": explain["std_SOS"]["DMSO"].values,
        "RAF": explain["std_RAF"]["DMSO"].values,
    })
    write_dat(out_8a, outdir / "Fig8a_std_DMSO.dat")
    exported["Fig8a"] = out_8a

    out_8b = pd.DataFrame({
        "Time_min": times,
        "SOS": explain["std_SOS"]["Drug"].values,
        "RAF": explain["std_RAF"]["Drug"].values,
    })
    write_dat(out_8b, outdir / "Fig8b_std_Drug.dat")
    exported["Fig8b"] = out_8b

    return exported

# ------------------------------------------------------------
# Export all
# ------------------------------------------------------------

def export_dat(data, stats, metrics, explain,
                          outdir="origin_dat",
                          rep_times_fig2=None,
                          rep_times_fig3=None,
                          rep_times_fig6=None,
                          bins_fig2=HIST_BINS,
                          bins_fig3=HIST_BINS):
    """
    Export all figure data into separate .dat files suitable for Origin.
    """
    if rep_times_fig2 is None:
        rep_times_fig2 = REP_TIMES
    if rep_times_fig3 is None:
        rep_times_fig3 = REP_TIMES
    if rep_times_fig6 is None:
        rep_times_fig6 = REP_TIMES1

    exported = {}

    exported["fig1"] = export_fig1_dat(data, outdir=outdir)
    exported["fig2"] = export_fig2_dat(data, rep_times=rep_times_fig2, outdir=outdir, bins=bins_fig2)
    exported["fig3"] = export_fig3_dat(data, rep_times=rep_times_fig3, outdir=outdir, bins=bins_fig3)
    exported["fig4"] = export_fig4_dat(stats, outdir=outdir)
    exported["fig5"] = export_fig5_dat(metrics, outdir=outdir)
    exported["fig6"] = export_fig6_dat(metrics, rep_times=rep_times_fig6, outdir=outdir)
    exported["fig7"] = export_fig7_dat(explain, outdir=outdir)
    exported["fig8"] = export_fig8_dat(explain, outdir=outdir)

    return exported

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    data = load_all_data(FILE_MAP)
    stats, metrics, explain = build_all_tables(data)
    
    # ---- export .dat files for Origin Pro ----
    exported = export_dat(
        data,
        stats,
        metrics,
        explain,
        outdir="Exported-Data-final",
        rep_times_fig2=REP_TIMES,
        rep_times_fig3=REP_TIMES,
        rep_times_fig6=REP_TIMES1,
        bins_fig2=HIST_BINS,
        bins_fig3=HIST_BINS
    )

    fig1, ax1 = plot_figure1(data)
    fig2, ax2 = plot_figure2(data, rep_times=REP_TIMES)
    fig3, ax3 = plot_figure3(data, rep_times=REP_TIMES)
    fig4, ax4 = plot_figure4(stats)
    fig5, ax5 = plot_figure5(metrics)
    fig6, ax6 = plot_figure6(metrics, rep_times=REP_TIMES1)
    fig7, ax7 = plot_figure7(explain)
    fig8, ax8 = plot_figure8(explain)

    plt.show()