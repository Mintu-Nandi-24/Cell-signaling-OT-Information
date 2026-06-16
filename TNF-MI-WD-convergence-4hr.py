"""
Convergence diagnostics for Mutual Information (MI) and 2-Wasserstein distance (2-WD)
====================================================================================

This script evaluates the numerical convergence of MI and 2-WD as a function of the
discretization density (ρ_Z) in the response variable (y). The analysis is restricted
to 4-hour datasets (WT and A20 backgrounds; NF-κB and ATF-2 responses).

Workflow
--------
1. Load experimental summary data (mean and std of y conditioned on x).
2. Construct Gaussian conditional response distributions p(y|x).
3. Optimize input distribution p(x) such that the resulting MI matches the
   experimentally reported target value.
4. Compute the marginal output distribution p(y).
5. Compute 2-WD between the input (x) and output (y) distributions using
   quantile mapping.
6. Check convergence by varying the discretization density (ρ_Z, called
   `points_per_std` in this code) and tracking the number of bins (n_Z).

Outputs
-------
- Diagnostic plots showing MI and 2-WD convergence across discretization densities.
- Exported convergence data for each dataset:
  * `{label}_pps-ny-W2.dat` containing:
    - points_per_std (ρ_Z) = Discretization density in the paper
    - n_y (number of y-grid bins, n_Z) = Discretized bins in the paper
    - 2-WD values

Dependencies
------------
- Python 3.8+
- numpy
- scipy
- matplotlib

Usage
-----

This will compute MI and 2-WD for all 4-hr datasets, display diagnostic plots,
and export the numerical results.

Author
------
[Mintu Nandi], [Universal Biology Institute, The University of Tokyo, 7-3-1 Hongo, Bunkyo-ku, Tokyo 113-0033, Japan]

"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize

# Reproducibility
np.random.seed(42)


# === Input files and target MI values ===
# Each entry: (filename, target_MI_bits)
filenames = [
    ('X-meanYcX-stdYcX-WT-4hr-nfkb.txt', 0.575),
    ('X-meanYcX-stdYcX-A20-4hr-nfkb.txt', 0.701),
    ('X-meanYcX-stdYcX-WT-4hr-atf-2.txt', 0.205),
    ('X-meanYcX-stdYcX-A20-4hr-atf-2.txt', 0.263)
]

# === Mutual Information computation ===
def compute_MI(px, py_given_x, dy):
    py = np.dot(px, py_given_x)
    px_tile = px[:, None]
    p_joint = px_tile * py_given_x
    p_prod = px_tile * py
    mi_density = p_joint * np.log2((p_joint + 1e-12) / (p_prod + 1e-12))
    return np.sum(mi_density) * dy

# === Compute 2-Wasserstein distance using quantile method ===
def compute_wasserstein2_1d(px, x_support, py, y_support, fname):
    sort_x = np.argsort(x_support)
    sort_y = np.argsort(y_support)
    x_vals_sorted = x_support[sort_x]
    y_vals_sorted = y_support[sort_y]
    
    px_sorted = px[sort_x]
    py_sorted = py[sort_y]
    
    # CDFs and quantiles
    cdf_x = np.cumsum(px_sorted)
    cdf_y = np.cumsum(py_sorted)
    quantiles = np.linspace(0, 1, 1000)
    x_quant = np.interp(quantiles, cdf_x, x_vals_sorted)
    y_quant = np.interp(quantiles, cdf_y, y_vals_sorted)
    
    # Dataset-specific conversion of y-axis into TNF conc. scale
    if fname == "X-meanYcX-stdYcX-WT-4hr-nfkb.txt":
        y_quant /= 215.016
    elif fname == "X-meanYcX-stdYcX-WT-4hr-atf2.txt":
        y_quant /= 156.863
    elif fname == "X-meanYcX-stdYcX-A20-4hr-nfkb.txt":
        y_quant /= 27.206
    elif fname == "X-meanYcX-stdYcX-A20-4hr-atf2.txt":
        y_quant /= 26.129
    else:
        y_quant /= 185  
    
    return np.sqrt(np.mean((x_quant - y_quant)**2))

# === Convergence resolution parameters ===
pps_list = np.linspace(100,500,50)

# === Store results ===
all_results = {}

# === Main loop: process each dataset ===
for fname, target_I in filenames:
    data = np.loadtxt(fname)
    x_vals = data[:, 0]
    mu_y = data[:, 1]
    std_y = data[:, 2]

    MI_vals = []
    W2_vals = []
    n_y_list = []
    for points_per_std in pps_list: #points_per_std=Discretization density (rho_Z in the paper)
        # Define y-grid based on current resolution
        y_min = max(0, np.min(mu_y - 4 * std_y))
        y_max = np.max(mu_y + 4 * std_y)
        min_std = np.min(std_y[std_y > 0])
        y_range = y_max - y_min
        n_y = int(np.ceil(points_per_std * y_range / min_std)) #Discretized bins (n_Z in the papar)
        y_grid = np.linspace(y_min, y_max, n_y)
        dy = y_grid[1] - y_grid[0]

        # Conditional distributions p(y|x)
        py_given_x = np.zeros((len(mu_y), len(y_grid)))
        for i, (mu, sigma) in enumerate(zip(mu_y, std_y)):
            sigma = max(sigma, 1e-8)
            py_given_x[i, :] = norm.pdf(y_grid, mu, sigma)
        py_given_x /= np.sum(py_given_x, axis=1, keepdims=True)

        # Optimize p(x) to match target MI
        def obj(px):
            px = np.abs(px)
            px /= np.sum(px)
            return (compute_MI(px, py_given_x, dy) - target_I)**2

        bounds = [(0, 1)] * len(mu_y)
        cons = {'type': 'eq', 'fun': lambda p: np.sum(p) - 1}
        px0 = np.ones(len(mu_y)) / len(mu_y)
        res = minimize(obj, px0, bounds=bounds, constraints=cons, method='SLSQP')
        px_opt = res.x / np.sum(res.x)

        # Marginal output p(y)
        py = py_given_x.T @ px_opt
        py /= np.sum(py)

        # Histogram approximation of p(y)
        bin_edges = np.linspace(np.min(y_grid), np.max(y_grid), 30)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        py_hist = np.histogram(np.repeat(y_grid, (py * 1e5).astype(int)), bins=bin_edges)[0]
        py_hist = py_hist + 1e-15
        py_hist /= np.sum(py_hist)
        
        # Compute MI
        MI = compute_MI(px_opt, py_given_x, dy)
        MI_vals.append(MI)

        # Compute 2-WD
        W2 = compute_wasserstein2_1d(px_opt, x_vals, py_hist, bin_centers, fname)
        W2_vals.append(W2)
        n_y_list.append(n_y)

    all_results[fname] = {'W2': W2_vals, 'MI': MI_vals, 'n_y': n_y_list}
    print(f"{fname}: 2-WD convergence done.")


epsilon = 1e-5 # convergence threshold

# === Plotting: 2-WD and MI convergence ===
fig, axs = plt.subplots(1, 2, figsize=(14, 10), sharex=True)

# Top-Right: 2-WD vs points_per_std
for key in all_results:
    label = key.replace('X-meanYcX-stdYcX-', '').replace('.txt', '')
    W2_vals = np.array(all_results[key]['W2'])
    axs[0, 1].plot(pps_list, W2_vals, 'o-', label=label)
    axs[0, 1].axhline(epsilon, color='k', linestyle='--', lw=1)
# axs[0, 1].set_yscale("log")
axs[0, 1].set_title("2-WD vs points_per_std (4hr)")
axs[0, 1].grid(True)
axs[0, 1].legend()

# Bottom-Right: n_y vs points_per_std
for key in all_results:
    label = key.replace('X-meanYcX-stdYcX-', '').replace('.txt', '')
    axs[1, 1].plot(pps_list, all_results[key]['n_y'], 'o-', label=label)
axs[1, 1].set_title("n_y vs points_per_std (4hr)")
axs[1, 1].set_xlabel("points_per_std")
axs[1, 1].grid(True)
axs[1, 1].legend()

plt.suptitle("2-WD Convergence Diagnostics and Discretization Mapping", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()




fig, axs = plt.subplots(1, 2, figsize=(14, 10), sharex=True)

# Top-Right: MI vs points_per_std
for key in all_results:
    label = key.replace('X-meanYcX-stdYcX-', '').replace('.txt', '')
    MI_vals = np.array(all_results[key]['MI'])
    axs[0, 1].plot(pps_list, MI_vals, 'o-', label=label)
axs[0, 1].set_title("MI vs points_per_std (4hr)")
axs[0, 1].grid(True)
axs[0, 1].legend()


# Bottom-Right: n_y vs points_per_std
for key in all_results:
    label = key.replace('X-meanYcX-stdYcX-', '').replace('.txt', '')
    axs[1, 1].plot(pps_list, all_results[key]['n_y'], 'o-', label=label)
axs[1, 1].set_title("n_y vs points_per_std (4hr)")
axs[1, 1].set_xlabel("points_per_std")
axs[1, 1].grid(True)
axs[1, 1].legend()

plt.suptitle("MI Convergence Diagnostics and Discretization Mapping", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()




import os

# === Export convergence data ===
outdir = 'MI_WD_exports'
os.makedirs(outdir, exist_ok=True)

# Export convergence data for each dataset
for fname, result in all_results.items():
    label = fname.replace('X-meanYcX-stdYcX-', '').replace('.txt', '')
    label_clean = label.replace(' ', '').replace('-', '')

    convergence_array = np.column_stack((pps_list, result['n_y'], result['W2']))
    np.savetxt(
        f'{outdir}/{label_clean}_pps-ny-W2.dat',
        convergence_array,
        header='points_per_std\tn_y\tW2',
        fmt='%.6e',
        delimiter='\t'
    )

print("Convergence files saved in 'MI_WD_exports/'")
