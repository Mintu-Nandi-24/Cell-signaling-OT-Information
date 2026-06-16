"""
Mutual Information (MI) and 2-Wasserstein Distance (2-WD) Analysis
==================================================================

This script computes the mutual information (MI) and 2-Wasserstein distance (2-WD)
between TNF dose (input, x) and responses (output, y) 
at 4-hour TNF stimulation (WT and A20 cells, NF-κB and ATF-2 responses). 

Workflow
--------
1. Load experimental summary data (mean and std of y conditioned on x).
2. Construct Gaussian conditional response distributions p(y|x).
3. Optimize input distribution p(x) to reproduce the experimentally reported MI.
4. Compute the output distribution p(y) and the joint distribution p(x,y).
5. Compute 2-WD between input (x) and output (y) distributions using quantile mapping.
6. Estimate uncertainty in MI and 2-WD using bootstrap resampling.
7. Generate figures:
   - Heatmaps of joint distributions with marginal histograms.
   - Scatter plot of MI vs 2-WD across all conditions.
   - CDF plots of input and output distributions.
8. Export numerical results and distributions for reproducibility.

Outputs
-------
All results are exported into the `MI_WD_exports/` directory:
- Conditional distributions: `*_py_given_x{i}.dat`
- Input distribution: `*_x-px.dat`
- Output distribution: `*_y-py.dat`
- Joint distribution: `*_x-y-pxy.dat`
- MI ± std and 2-WD ± std: `*_MI+-std-2WD+-std.dat`
- Input and output CDFs: `*_x-cdfx.dat`, `*_y-cdfy.dat`
- Quantile functions: `*_q-xquant.dat`, `*_q-yquant.dat`


Dependencies
------------
- Python 3.8+
- numpy
- scipy
- matplotlib

Usage
-----
Run the script directly:

    python MI_2WD_analysis_30min.py

This will compute all quantities, show figures, and export numerical data.

Author
------
[Mintu Nandi], [Universal Biology Institute, The University of Tokyo, 7-3-1 Hongo, Bunkyo-ku, Tokyo 113-0033, Japan]

"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# Set random seed for reproducibility
np.random.seed(42)

# === Load datasets (only 4 hr data, 30 min data removed) ===
# Format: (filename, target_mutual_information)
filenames = [
    ('X-meanYcX-stdYcX-WT-30min-nfkb.txt', 0.918),
    ('X-meanYcX-stdYcX-A20-30min-nfkb.txt', 0.423),
    ('X-meanYcX-stdYcX-WT-30min-atf-2.txt', 0.854),
    ('X-meanYcX-stdYcX-A20-30min-atf-2.txt', 0.136)
]

# === Bootstrap Error Estimation ===
def bootstrap_MI_WD(x_vals, mu_y, std_y, bin_edges, bin_centers, B=100):
    """
    Estimate standard deviations of MI and WD using bootstrap resampling.
    """
    MI_boot = []
    WD_boot = []
    N = len(x_vals)

    for b in range(B):
        try:
            indices = np.random.choice(N, N, replace=True)
            x_sample = x_vals[indices]
            mu_sample = mu_y[indices]
            std_sample = std_y[indices]

            y_min = max(0, np.min(mu_sample - 4 * std_sample))
            y_max = np.max(mu_sample + 4 * std_sample)
            points_per_std = 200
            min_std = np.min(std_sample[std_sample > 0])
            y_range = y_max - y_min
            n_y = int(np.ceil(points_per_std * y_range / min_std))
            y_grid = np.linspace(y_min, y_max, n_y)
            dy = y_grid[1] - y_grid[0]

            py_given_x = np.zeros((len(mu_sample), len(y_grid)))
            for i, (mu, sigma) in enumerate(zip(mu_sample, std_sample)):
                sigma = max(sigma, 1e-8)
                py_given_x[i, :] = norm.pdf(y_grid, mu, sigma)
            py_given_x /= np.sum(py_given_x, axis=1, keepdims=True)

            px = np.ones(len(mu_sample)) / len(mu_sample)
            py = py_given_x.T @ px
            MI = compute_MI(px, py_given_x, dy)

            py_hist = np.histogram(np.repeat(y_grid, (py * 1e5).astype(int)), bins=bin_edges)[0]
            if np.sum(py_hist) == 0:
                continue  # Skip this bootstrap replicate
            py_hist = py_hist + 1e-15
            py_hist /= np.sum(py_hist)
            W2, *_ = compute_wasserstein2_1d(px, x_sample, py_hist, bin_centers)

            if not np.isfinite(MI) or not np.isfinite(W2):
                continue
            
            MI_boot.append(MI)
            WD_boot.append(W2)
        except:
            continue

    MI_std = np.std(MI_boot, ddof=1)
    WD_std = np.std(WD_boot, ddof=1)
    return MI_std, WD_std


# === Mutual Information computation ===
def compute_MI(px, py_given_x, dy):
    py = np.dot(px, py_given_x)
    px_tile = px[:, None]
    p_joint = px_tile * py_given_x
    p_prod = px_tile * py
    mi_density = p_joint * np.log2((p_joint + 1e-12) / (p_prod + 1e-12))
    return np.sum(mi_density) * dy

# === Optimization to match target MI ===
def match_target_mi(mu_y, std_y, target_I):
    """
    Find input distribution p(x) that yields a given target mutual information.
    """
    y_min = max(0, np.min(mu_y - 4 * std_y))
    y_max = np.max(mu_y + 4 * std_y)
    
    points_per_std = 200 # Discretization density (rho_Z in the paper)
    min_std = np.min(std_y[std_y > 0])  
    y_range = y_max - y_min
    n_y = int(np.ceil(points_per_std * y_range / min_std)) # Discretized bins (n_Z in the paper)
      
    
    y_grid = np.linspace(y_min, y_max, n_y)# 5000)
    dy = y_grid[1] - y_grid[0]

    # Conditional distributions p(y|x)
    py_given_x = np.zeros((len(mu_y), len(y_grid)))
    for i, (mu, sigma) in enumerate(zip(mu_y, std_y)):
        py_given_x[i, :] = norm.pdf(y_grid, mu, sigma)
    py_given_x /= np.sum(py_given_x, axis=1, keepdims=True)

    # Objective: match target mutual information
    def obj(px):
        px = np.abs(px)
        px /= np.sum(px)
        I = compute_MI(px, py_given_x, dy)
        return (I - target_I)**2

    cons = ({'type': 'eq', 'fun': lambda p: np.sum(p) - 1})
    bounds = [(0, 1) for _ in mu_y]
    px0 = np.ones(len(mu_y)) / len(mu_y)
    res = minimize(obj, px0, method='SLSQP', bounds=bounds, constraints=cons)

    px_opt = res.x / np.sum(res.x)
    matched_I = compute_MI(px_opt, py_given_x, dy)
    py = py_given_x.T @ px_opt
    return px_opt, py, y_grid, matched_I, py_given_x

# === 2-Wasserstein distance calculation ===
def compute_wasserstein2_1d(px, x_support, py, y_support):
    """
    Compute 1D Wasserstein-2 distance between input and output distributions.
    """
    sort_x = np.argsort(x_support)
    sort_y = np.argsort(y_support)
    x_vals_sorted = x_support[sort_x]
    y_vals_sorted = y_support[sort_y]
    px_sorted = px[sort_x]
    py_sorted = py[sort_y]
    cdf_x = np.cumsum(px_sorted)
    cdf_y = np.cumsum(py_sorted)
    quantiles = np.linspace(0, 1, 1000)
    x_quant = np.interp(quantiles, cdf_x, x_vals_sorted)
    y_quant = np.interp(quantiles, cdf_y, y_vals_sorted)
    
    # Dataset-specific conversion of y-axis into TNF conc. scale
    if fname == "X-meanYcX-stdYcX-WT-30min-nfkb.txt":
        y_quant_norm = y_quant / 1408.232
    elif fname == "X-meanYcX-stdYcX-WT-30min-atf2.txt":
        y_quant_norm = y_quant / 3975.744
    elif fname == "X-meanYcX-stdYcX-A20-30min-nfkb.txt":
        y_quant_norm = y_quant / 1033.785
    elif fname == "X-meanYcX-stdYcX-A20-30min-atf2.txt":
        y_quant_norm = y_quant / 2934.247
    else:
        y_quant_norm = y_quant / 185    
    
    W2_squared = np.mean((x_quant - y_quant_norm)**2)
    W2 = np.sqrt(W2_squared)
    
    return W2, quantiles, x_quant, y_quant, cdf_x, cdf_y, x_vals_sorted, y_vals_sorted


# === Figure Generation ===
fig = plt.figure(figsize=(14, 20))
outer_gs = GridSpec(2, 2, figure=fig)

MI_vals, WD_vals, labels = [], [], []
x_vals_all, px_all, y_vals_all, py_all = [], [], [], []

for idx, (fname, target_I) in enumerate(filenames):
    data = np.loadtxt(fname)
    x_vals = data[:, 0]
    mu_y = data[:, 1]
    std_y = data[:, 2]

    # Optimize input distribution to match target MI
    px_opt, py, y_grid, matched_I, py_given_x = match_target_mi(mu_y, std_y, target_I)
    p_joint = px_opt[:, None] * py_given_x

    # Grid layout for joint + marginals
    row, col = divmod(idx, 2)
    inner_gs = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_gs[row, col],
                                       width_ratios=[5, 1], height_ratios=[1, 5],
                                       wspace=0.05, hspace=0.05)
    ax_top = fig.add_subplot(inner_gs[0, 0])
    ax_right = fig.add_subplot(inner_gs[1, 1])
    ax_joint = fig.add_subplot(inner_gs[1, 0])

    # Heatmap of joint distribution p(x,y)
    im = ax_joint.imshow(p_joint, aspect='auto', origin='lower',
                         extent=[y_grid[0], y_grid[-1], x_vals[0], x_vals[-1]],
                         norm=LogNorm(), cmap='viridis')
    ax_joint.set_xlabel("y (response)")
    ax_joint.set_ylabel("x (TNF dose)")
    ax_joint.set_title(f"{fname}\nMI = {matched_I:.3f} bits", fontsize=10)

    # Output distribution p(y)
    bin_edges = np.linspace(np.min(y_grid), np.max(y_grid), 30)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    p_y_hist = np.histogram(np.repeat(y_grid, (py * 100000).astype(int)), bins=bin_edges)[0]
    ax_top.bar(bin_centers, p_y_hist, width=np.diff(bin_edges), color='gray', edgecolor='black')
    ax_top.set_xlim(y_grid[0], y_grid[-1])
    ax_top.set_ylabel("p(y)")
    ax_top.tick_params(labelbottom=False)

    # Input distribution p(x)
    ax_right.barh(x_vals, px_opt)
    ax_right.set_xscale("log")
    ax_right.set_yscale("log")
    ax_right.set_xlabel("p(x)")
    ax_right.tick_params(labelleft=False)

    # Colorbar for joint distribution
    cbar = fig.colorbar(im, ax=ax_joint, orientation='vertical', shrink=0.6)
    cbar.ax.set_ylabel('p(x, y)', rotation=270, labelpad=10)

    # 2-Wasserstein distance computation
    x_support = x_vals
    px = px_opt / np.sum(px_opt)
    y_support = bin_centers
    py = p_y_hist + 1e-15
    py /= np.sum(py)
    W2, quantiles, x_quant, y_quant, cdf_x, cdf_y, x_sorted, y_sorted = compute_wasserstein2_1d(
        px, x_support, py, y_support)
    
    # === Bootstrap error estimation ===
    MI_std, W2_std = bootstrap_MI_WD(x_vals, mu_y, std_y, bin_edges, bin_centers, B=100)

    # Collect results
    MI_vals.append(matched_I)
    WD_vals.append(W2)
    label = fname.replace('X-meanYcX-stdYcX-', '').replace('.txt', '')
    labels.append(label)
    x_vals_all.append(x_vals.copy())
    px_all.append(px.copy())
    y_vals_all.append(bin_centers.copy())
    py_all.append(py.copy())

    print(f"{fname}: matched MI = {matched_I:.4f} bits, Wasserstein-2 = {W2:.4f}")

    import os

    # Create output directory
    outdir = 'MI_WD_exports'
    os.makedirs(outdir, exist_ok=True)

    label_clean = label.replace(' ', '').replace('-', '')
    
    # === Export p(y|x) ===
    for i, x_val in enumerate(x_vals):
        fname = f'{outdir}/{label_clean}_py_given_x{i+1}.dat'
        header = f'# p(y | x = {x_val:.6e})\n# y_val\tp(y|x)'
        np.savetxt(
            fname,
            np.column_stack((y_grid, py_given_x[i])),
            header=header,
            fmt='%.6e',
            delimiter='\t'
            )


    # === Export p(x) ===
    np.savetxt(
        f'{outdir}/{label_clean}_x-px.dat',
        np.column_stack((x_vals, px_opt)),
        header='x_val\tp(x)',
        fmt='%.6e',
        delimiter='\t'
        )

    # === Export p(y) ===
    np.savetxt(
        f'{outdir}/{label_clean}_y-py.dat',
        np.column_stack((bin_centers, py)),
        header='y_val\tp(y)',
        fmt='%.6e',
        delimiter='\t'
        )

    # === Export p(x, y) ===
    X, Y = np.meshgrid(y_grid, x_vals, indexing='xy')  # Same indexing as imshow
    xy_pxy = np.column_stack((Y.ravel(), X.ravel(), p_joint.ravel()))
    np.savetxt(
        f'{outdir}/{label_clean}_x-y-pxy.dat',
        xy_pxy,
        header='x_val\ty_val\tp(x,y)',
        fmt='%.6e',
        delimiter='\t'
        )

    # === Export MI ± std and WD ± std ===
    np.savetxt(
        f'{outdir}/{label_clean}_MI+-std-WD+-std.dat',
        [[matched_I, MI_std, W2, W2_std]],
        header='MI\tMI_std\tWD\tWD_std',
        fmt='%.6e',
        delimiter='\t'
        )
    
    
    # === Export CDFs ===
    np.savetxt(f'{outdir}/{label_clean}_x-cdfx.dat',
               np.column_stack((x_sorted, cdf_x)),
               header='x_val\tCDF(x)', fmt='%.6e', delimiter='\t')
    np.savetxt(f'{outdir}/{label_clean}_y-cdfy.dat',
               np.column_stack((y_sorted, cdf_y)),
               header='y_val\tCDF(y)', fmt='%.6e', delimiter='\t')

    # === Export quantile functions ===
    np.savetxt(f'{outdir}/{label_clean}_q-xquant.dat',
               np.column_stack((quantiles, x_quant)),
               header='q\tx_q', fmt='%.6e', delimiter='\t')
    np.savetxt(f'{outdir}/{label_clean}_q-yquant.dat',
               np.column_stack((quantiles, y_quant)),
               header='q\ty_q', fmt='%.6e', delimiter='\t') 

plt.show()

# === Summary Scatter Plot: MI vs WD ===
plt.figure(figsize=(6, 5))
colors = ['blue', 'red', 'green', 'orange']
for i in range(len(MI_vals)):
    plt.scatter(WD_vals[i], MI_vals[i], color=colors[i], label=labels[i], s=100, edgecolor='black')
    plt.annotate(labels[i], (WD_vals[i], MI_vals[i]), textcoords="offset points", xytext=(5, 5), fontsize=9)
plt.xlabel('Wasserstein-2 Distance')
plt.ylabel('Mutual Information (bits)')
plt.title('MI vs WD Across Conditions')
plt.grid(True)
plt.tight_layout()
plt.show()

# === CDF Figure for input and output distributions ===
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
for i in range(len(labels)):
    x_sorted = np.array(x_vals_all[i])
    px_sorted = np.array(px_all[i])
    sort_idx_x = np.argsort(x_sorted)
    axs[0].plot(x_sorted[sort_idx_x], np.cumsum(px_sorted[sort_idx_x]),
                lw=2, color=colors[i], label=f'{labels[i]}')
axs[0].set_xlabel('x (Signal)')
axs[0].set_ylabel('Cumulative Probability')
axs[0].set_title('CDF of Input Distributions $p(x)$')
axs[0].set_xscale('log')
axs[0].grid(True)
axs[0].legend()

for i in range(len(labels)):
    y_sorted = np.array(y_vals_all[i])
    py_sorted = np.array(py_all[i])
    sort_idx_y = np.argsort(y_sorted)
    axs[1].plot(y_sorted[sort_idx_y], np.cumsum(py_sorted[sort_idx_y]),
                lw=2, linestyle='-', color=colors[i], label=f'{labels[i]}')
axs[1].set_xlabel('y (Response)')
axs[1].set_title('CDF of Output Distributions $p(y)$')
axs[1].grid(True)
axs[1].legend()

plt.suptitle('CDFs of Input and Output Distributions Across Conditions', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
