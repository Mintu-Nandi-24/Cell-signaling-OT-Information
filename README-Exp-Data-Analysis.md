# TNF Signaling: Mutual Information and 2-Wasserstein Distance Estimation

This repository contains Python scripts for analyzing **Mutual Information (MI)** and **2-Wasserstein Distance (WD)** in TNF signaling networks, based on NF-κB and ATF-2 responses in WT and A20-deficient cells. The workflow combines statistical inference, optimization, and convergence diagnostics to quantify information–geometry trade-offs in cellular signaling.

---

## Contents

### 1. `MI-WD-estimation-Exp-Data.py`
**Purpose:**  
Computes MI and 2-WD between TNF dose (input) and cellular response (output) using experimental summary data (conditional means and standard deviations).  

**Key Features:**  
- Constructs Gaussian conditional response distributions \( P(y|x) \).  
- Optimizes input distribution \( P(x) \) to match experimentally reported MI.  
- Computes marginal \( P(y) \), joint \( P(x,y) \), and 2-WD via quantile mapping.  
- Estimates uncertainty via bootstrap resampling. 
- Exports all numerical distributions and diagnostics.  

**Output Files:**  
- `*_py_given_x{i}.dat` (conditional distributions)  
- `*_x-px.dat`, `*_y-py.dat` (marginals)  
- `*_x-y-pxy.dat` (joint)  
- `*_MI+-std-2WD+-std.dat` (MI and 2-WD with errors)  
- `*_x-cdfx.dat`, `*_y-cdfy.dat` (CDFs)  
- `*_q-xquant.dat`, `*_q-yquant.dat` (quantile functions)  

---

### 2. `MI-WD-convergence.py`
**Purpose:**  
Evaluates numerical convergence of MI and 2-WD as a function of discretization density (ρ_Z) in the response variable.  

**Key Features:**  
- Varies grid resolution (`points_per_std` = discretization density) for conditional distributions.  
- Optimizes input distribution to reproduce target MI.  
- Tracks convergence of MI and 2-WD with respect to `points_per_std`.  
- Plots MI and 2-WD vs resolution, and also plots discretized bins \( n_Z \).  
- Exports convergence diagnostics.  

**Output Files:**  
- `{label}_pps-ny-W2.dat` with:  
  - `points_per_std` (ρ_Z)  
  - `n_y` (discretized bins, n_Z)  
  - `2-WD values`  

---

### 3. `mapping-factor-estimation.py`
**Purpose:**  
Estimates mapping (conversion) factors between TNF dose and cellular response.  

**Key Features:**  
- For **WT** datasets: fits Hill functions and extracts parameters (Rmin, Rmax, EC50, Hill coefficient n). Conversion factor = slope at EC50.  
- For **A20-deficient** datasets:  
  - Primary model = linear or flat (chosen by R²).  
  - Secondary (comparison only) = Hill fit.  
- Generates plots of raw data with error bars and fitted curves.  
- Exports parameter tables and fitted curves.  

**Output Files:**  
- `WT-hill-fit-param-nfkb-aft2.dat` (Hill fit parameters for WT).  
- `A20-hill-fit-param-nfkb-aft2.dat` (Hill comparison for A20).  
- `A20-linear-fit-param-nfkb-aft2.dat` (linear/flat fits for A20).  
- `{label}_fit-curve.dat` (primary fits).  
- `{label}_hill-fit.dat` (Hill comparison for A20).  

---

## Dependencies

- Python 3.8+  
- numpy  
- scipy  
- pandas  
- scikit-learn  
- matplotlib  

Install with:  

```bash
pip install numpy scipy pandas scikit-learn matplotlib
```

---

## Usage

Run any script directly, e.g.:

```bash
python MI-WD-estimation-Exp-Data.py
python MI-WD-convergence.py
python mapping-factor-estimation.py
```

Each script will display figures and export results.

---

## Author

**Mintu Nandi**  
Universal Biology Institute, The University of Tokyo  
7-3-1 Hongo, Bunkyo-ku, Tokyo 113-0033, Japan  
