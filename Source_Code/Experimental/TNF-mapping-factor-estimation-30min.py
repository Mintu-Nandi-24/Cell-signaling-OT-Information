"""
Mapping Factor Estimation: Hill vs Linear/Flat Models
=====================================================

This script estimates mapping (conversion) factors for TNF dose–response data
using either Hill fits (for WT datasets) or linear/flat fits (for A20 datasets).
Hill fits are also computed for A20 datasets for comparison only.

Workflow
--------
1. Load 4-hour datasets (WT and A20, NF-κB and ATF-2 responses).
2. For WT datasets:
   - Fit a Hill function: Rmin + (Rmax - Rmin) / (1 + (EC50/dose)^n).
   - Extract parameters (Rmin, Rmax, EC50, Hill coefficient n).
   - Compute conversion factor from slope at EC50.
3. For A20 datasets:
   - Primary fit: linear or flat model chosen based on R².
   - Secondary (comparison only): Hill fit performed but not used in conversion.
   - Compute intercept, slope, R², residual sum of squares (RSS).
   - Use slope (if linear) or flat value as conversion factor.
4. Generate diagnostic plots:
   - Data with error bars.
   - Overlay fitted curves (Hill, linear, flat).
   - For A20: overlay both primary fit and Hill comparison fit.
5. Create summary tables:
   - Hill fit parameters for WT.
   - Hill fit comparison parameters for A20.
   - Linear/flat fit parameters for A20.
6. Export numerical results:
   - Tables of fit parameters and conversion factors.
   - Fitted response curves for all datasets.

Outputs
-------
Results are written to the directory `MI_WD_exports/`:
- `WT-hill-fit-param-nfkb-aft2.dat` → Hill fit parameters (WT).
- `A20-hill-fit-param-nfkb-aft2.dat` → Hill comparison parameters (A20).
- `A20-linear-fit-param-nfkb-aft2.dat` → Linear/flat fit parameters (A20).
- `{label}_fit-curve.dat` → primary fitted curves (WT Hill, A20 linear/flat).
- `{label}_hill-fit.dat` → A20 Hill comparison curves.

Figures are shown interactively.

Dependencies
------------
- Python 3.8+
- numpy
- pandas
- scipy
- scikit-learn
- matplotlib

Usage
-----
Run the script directly:

    python mapping-factor-estimation.py

This will perform all fits, generate plots, print summary tables,
and export results to `MI_WD_exports/`.

Author
------
[Mintu Nandi], [Universal Biology Institute, The University of Tokyo, 7-3-1 Hongo, Bunkyo-ku, Tokyo 113-0033, Japan]

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import warnings

# === Hill function ===
def hill_equation(dose, Rmin, Rmax, EC50, n):
    """
    Standard Hill equation for dose–response fitting.
    Rmin : minimum response
    Rmax : maximum response
    EC50 : half-maximal effective concentration
    n    : Hill coefficient (slope parameter)
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        dose = np.maximum(dose, 1e-12)
        return Rmin + (Rmax - Rmin) / (1 + (EC50 / dose)**n)

# === Hill fitting ===
def fit_hill(dose, response):
    """
    Fit Hill equation to data using nonlinear least squares.
    Returns best-fit parameters, R², and residual sum of squares.
    """
    dose = np.array(dose)
    response = np.array(response)
    valid = dose > 0
    dose = dose[valid]
    response = response[:len(dose)]

    Rmin_init = np.min(response)
    Rmax_init = np.max(response)
    EC50_init = dose[np.argmax(np.diff(response, prepend=0))]
    p0 = [Rmin_init, Rmax_init, EC50_init, 1.0]
    bounds = ([0, 0, 1e-6, 0.1], [2 * Rmax_init, 2 * Rmax_init, 1e3, 5])

    try:
        popt, _ = curve_fit(hill_equation, dose, response, p0=p0, bounds=bounds, maxfev=10000)
        pred = hill_equation(dose, *popt)
        ss_res = np.sum((response - pred) ** 2)
        ss_tot = np.sum((response - np.mean(response)) ** 2)
        r2 = 1 - ss_res / ss_tot
    except:
        popt = [np.nan]*4
        ss_res = np.nan
        r2 = np.nan

    return popt, r2, ss_res

# === Linear or flat fallback ===
def fit_linear_or_flat(dose, response):
    """
    For A20 datasets: fit either a linear model or a flat (constant) model.
    Select whichever gives higher R².
    """
    
    dose = dose[dose > 0]
    response = response[:len(dose)]

    A = np.vstack([dose, np.ones_like(dose)]).T
    lin_coef, *_ = np.linalg.lstsq(A, response, rcond=None)
    lin_pred = lin_coef[0] * dose + lin_coef[1]
    lin_r2 = r2_score(response, lin_pred)

    flat_pred = np.full_like(response, np.mean(response))
    flat_r2 = r2_score(response, flat_pred)

    if lin_r2 >= flat_r2:
        return lin_coef[1], lin_coef[0], lin_r2, np.sum((response - lin_pred) ** 2), 'linear'
    else:
        mean_val = np.mean(response)
        return mean_val, 0.0, flat_r2, np.sum((response - flat_pred) ** 2), 'flat'

# === Filepaths ===
filepaths = {
    "WT_30min_nfkb": "X-meanYcX-stdYcX-WT-30min-nfkb.txt",
    "A20_30min_nfkb": "X-meanYcX-stdYcX-A20-30min-nfkb.txt",
    "WT_30min_atf2": "X-meanYcX-stdYcX-WT-30min-atf-2.txt",
    "A20_30min_atf2": "X-meanYcX-stdYcX-A20-30min-atf-2.txt"
}

# === Storage containers ===
fitted_params = {}
conversion_factors = {}
r2_values = {}
residual_sums = {}
model_used = {}
slope_summary = {}
hill_fit_a20 = {}

fit_export_arrays = {}
fit_hill_a20_export_arrays = {}

# === Fitting and plotting ===
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()

for i, (label, path) in enumerate(filepaths.items()):
    data = pd.read_csv(path, sep=r"\s+", header=None, names=["Dose", "Mean", "Std"])
    dose = data["Dose"].values
    mean = data["Mean"].values
    std = data["Std"].values

    ax = axs[i]
    ax.errorbar(dose, mean, yerr=std, fmt='o', label='Data')
    ax.set_xscale('log')
    ax.set_xlabel("TNF (ng/mL)")
    ax.set_ylabel("Response (a.u.)")
    ax.grid(True, which="both", linestyle="--")

    if "A20" in label:
        # Secondary model: Hill fit (for comparison only)
        popt_hill, _, _ = fit_hill(dose, mean)
        hill_fit_a20[label] = popt_hill
        
        # Primary model: linear or flat
        intercept, slope, r2, rss, model = fit_linear_or_flat(dose, mean)
        fitted_params[label] = [intercept, intercept + slope * np.max(dose), np.nan, np.nan]
        conversion_factors[label] = slope
        slope_summary[label] = slope
        r2_values[label] = r2
        residual_sums[label] = rss
        model_used[label] = model

        if model == 'linear':
            ax.plot(dose, slope * dose + intercept, 'g--', label='Linear Fit')
            fit_vals = slope * dose + intercept
        else:
            ax.hlines(intercept, np.min(dose), np.max(dose), color='g', linestyle='--', label='Flat Fit')
            fit_vals = np.full_like(dose, intercept)
            
         # === Store primary fit ===
        fit_export_arrays[label] = np.column_stack([dose, fit_vals])


        # Overlay Hill fit
        popt = hill_fit_a20[label]
        if not np.isnan(popt).any():
            dose_range = np.logspace(np.log10(np.min(dose[dose > 0])), np.log10(np.max(dose)), 200)
            hill_vals = hill_equation(dose_range, *popt)
            ax.plot(dose_range, hill_vals, 'r:', label='Hill Fit')

            # === Store secondary A20 Hill fit ===
            fit_hill_a20_export_arrays[label] = np.column_stack([dose_range, hill_vals])
            
            

    else:
        # WT: use Hill fit
        popt, r2, rss = fit_hill(dose, mean)
        fitted_params[label] = popt
        Rmin, Rmax, EC50, n = popt
        slope = (Rmax - Rmin) * n / (4 * EC50)
        conversion_factors[label] = slope
        slope_summary[label] = np.nan
        r2_values[label] = r2
        residual_sums[label] = rss
        model_used[label] = 'hill'
        dose_range = np.logspace(np.log10(np.min(dose[dose > 0])), np.log10(np.max(dose)), 200)
        fit_vals = hill_equation(dose_range, *popt)
        ax.plot(dose_range, fit_vals, 'r--', label='Hill Fit')
        
        # === Store WT Hill fit ===
        fit_export_arrays[label] = np.column_stack([dose_range, fit_vals])

        
    ax.set_title(f"{label} ({model_used[label]})")
    ax.legend()

plt.tight_layout()
plt.show()

# === WT Hill Table ===
wt_rows = []
for label in filepaths:
    if "WT" in label:
        Rmin, Rmax, EC50, n = fitted_params[label]
        conv = conversion_factors[label]
        wt_rows.append([label, Rmin, Rmax, EC50, n, conv])

wt_table = pd.DataFrame(wt_rows, columns=["Condition", "Rmin", "Rmax", "EC50", "Hill_n", "Conversion Factor"])
print("\n=== Hill Fit Parameters (WT) ===")
print(wt_table.to_string(index=False, float_format="%.6f"))

# === A20 Hill Comparison Table ===
a20_hill_rows = []
for label in filepaths:
    if "A20" in label:
        popt = hill_fit_a20[label]
        if not np.isnan(popt).any():
            Rmin, Rmax, EC50, n = popt
            conv = (Rmax - Rmin) * n / (4 * EC50)
        else:
            Rmin, Rmax, EC50, n, conv = [np.nan]*5
        a20_hill_rows.append([label, Rmin, Rmax, EC50, n, conv])

a20_hill_table = pd.DataFrame(a20_hill_rows, columns=[
    "Condition", "Rmin", "Rmax", "EC50", "Hill_n", "Conversion Factor"
])

print("\n=== Hill Fit Parameters (A20, For Comparison Only) ===")
print(a20_hill_table.to_string(index=False, float_format="%.6f"))


# === A20 Linear/Flat Table ===
a20_rows = []
for label in filepaths:
    if "A20" in label:
        intercept = fitted_params[label][0]
        slope = slope_summary[label]
        r2 = r2_values[label]
        rss = residual_sums[label]
        model = model_used[label]
        a20_rows.append([label, model, intercept, slope, r2, rss, slope])

a20_table = pd.DataFrame(a20_rows, columns=[
    "Condition", "Model", "Intercept", "Slope", "R²", "RSS", "Conversion Factor"
])
print("\n=== Linear/Flat Fit Parameters and Conversion Factors (A20) ===")
print(a20_table.to_string(index=False, float_format="%.6f"))



import os

# === Create Export Directory ===
export_dir = "MI_WD_exports"
os.makedirs(export_dir, exist_ok=True)

# === Export Tables as .dat ===
# WT table (exclude label)
np.savetxt(os.path.join(export_dir, "WT-hill-fit-param-nfkb-aft2.dat"),
           wt_table.iloc[:, 1:].values,
           fmt="%.6f",
           delimiter="\t",
           header="\t".join(wt_table.columns[1:]),
           comments='')

# A20 Hill comparison table (exclude label)
np.savetxt(os.path.join(export_dir, "A20-hill-fit-param-nfkb-aft2.dat"),
           a20_hill_table.iloc[:, 1:].values,
           fmt="%.6f",
           delimiter="\t",
           header="\t".join(a20_hill_table.columns[1:]),
           comments='')

# A20 linear/flat fit table (exclude label and model name)
np.savetxt(os.path.join(export_dir, "A20-linear-fit-param-nfkb-aft2.dat"),
           a20_table.iloc[:, 2:].values,
           fmt="%.6f",
           delimiter="\t",
           header="\t".join(a20_table.columns[2:]),
           comments='')

# === Export Fitted Curves (Primary and Hill Comparison) ===
for label, arr in fit_export_arrays.items():
    np.savetxt(os.path.join(export_dir, f"{label}_fit-curve.dat"),
               arr,
               fmt="%.6f",
               delimiter="\t",
               header="Dose\tFit_Response",
               comments='')

for label, arr in fit_hill_a20_export_arrays.items():
    np.savetxt(os.path.join(export_dir, f"{label}_hill-fit.dat"),
               arr,
               fmt="%.6f",
               delimiter="\t",
               header="Dose\tHill_Comparison_Response",
               comments='')

