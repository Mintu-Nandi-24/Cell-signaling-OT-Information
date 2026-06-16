# Experimental Data Analysis: TNF Signaling and RAS-MAPK SOS-RAF Signaling

This directory contains Python scripts for analyzing **mutual information (MI)** and **2-Wasserstein distance (2-WD)** in experimental signaling data. The TNF scripts analyze NF-κB and ATF-2 responses in WT and A20-deficient cells, while the RAS-MAPK script analyzes SOS and RAF membrane-translocation dynamics under control and MEK-inhibited conditions.

---

## Contents

## 1. TNF Signaling Analysis

The TNF workflow is applied separately to the **4-hour** and **30-minute** response regimes. The suffix in each filename indicates the corresponding time point:

- `-4hr.py` scripts analyze the late 4-hour TNF response.
- `-30min.py` scripts analyze the early 30-minute TNF response.

### 1.1 MI and 2-WD estimation

- **`MI-WD-estimation-Exp-Data-4hr.py`** – Computes MI and 2-WD for the 4-hour TNF response.
- **`MI-WD-estimation-Exp-Data-30min.py`** – Computes MI and 2-WD for the 30-minute TNF response.

**Purpose:**  
Computes MI and 2-WD between TNF dose (input) and cellular response (output) using experimental summary data, including conditional means and standard deviations.

**Key features:**
- Constructs Gaussian conditional response distributions \(p(y|x)\).
- Optimizes the input distribution \(p(x)\) to match experimentally reported MI.
- Computes marginal \(p(y)\), joint \(p(x,y)\), and 2-WD via quantile mapping.
- Estimates uncertainty by bootstrap resampling.
- Exports conditional distributions, marginals, joint distributions, CDFs, quantile functions, MI, and 2-WD values.

### 1.2 Numerical convergence analysis

- **`MI-WD-convergence-4hr.py`** – Performs convergence analysis for the 4-hour TNF response.
- **`MI-WD-convergence-30min.py`** – Performs convergence analysis for the 30-minute TNF response.

**Purpose:**  
Evaluates numerical convergence of MI and 2-WD as a function of discretization density in the response variable.

**Key features:**
- Varies grid resolution for conditional response distributions.
- Tracks convergence of MI and 2-WD with respect to discretization density.
- Exports convergence diagnostics to `MI_WD_exports/`.

### 1.3 Mapping-factor estimation

- **`mapping-factor-estimation-4hr.py`** – Estimates mapping factors for the 4-hour TNF response.
- **`mapping-factor-estimation-30min.py`** – Estimates mapping factors for the 30-minute TNF response.

**Purpose:**  
Estimates mapping or conversion factors between TNF dose and cellular response.

**Key features:**
- For WT datasets, fits Hill functions and extracts response parameters.
- For A20-deficient datasets, uses a linear or flat primary model based on model fit.
- Exports fitted parameters and fitted response curves.

---

## 2. RAS-MAPK SOS-RAF Analysis

### `SOS-RAF-analysis.py`

**Purpose:**  
Analyzes published single-cell SOS and RAF membrane-translocation trajectories under control (DMSO) and MEK-inhibited (MEKi/Drug) conditions.

**Correspondence with the dual-fidelity framework:**
- SOS is treated as the input variable \(X\).
- RAF is treated as the output variable \(Z\).
- INF is computed as Gaussian MI from the SOS-RAF correlation coefficient.
- GMF is computed as the inverse 1D Gaussian 2-Wasserstein distance between SOS and RAF marginal distributions.

**Expected input files:**
- `SOS_wt_DMSO_EGF10ng.csv`
- `RAF_wt_DMSO_EGF10ng.csv`
- `SOS_wt_10nMMEKi_EGF10ng.csv`
- `RAF_wt_10nMMEKi_EGF10ng.csv`

**Key features:**
- Loads matched SOS and RAF single-cell trajectories.
- Computes time-dependent mean, standard deviation, coefficient of variation, dynamic range, and SOS-RAF correlation.
- Computes Gaussian MI, 2-WD, and inverse 2-WD at each time point.
- Estimates uncertainty by paired bootstrap resampling of matched SOS-RAF cell pairs.
- Exports processed figure data to `Exported-Data-final/` for plotting in OriginPro.

---

## 3. Output Directories

Depending on the script, output files are written to:

- `MI_WD_exports/` for TNF MI/2-WD estimation, convergence, and mapping-factor analyses.
- `Exported-Data-final/` for RAS-MAPK SOS-RAF time-series, distributional, and dual-fidelity analyses.

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

Run each script directly from the folder containing the required input data.

### TNF analysis

```bash
python MI-WD-estimation-Exp-Data-4hr.py
python MI-WD-estimation-Exp-Data-30min.py
python MI-WD-convergence-4hr.py
python MI-WD-convergence-30min.py
python mapping-factor-estimation-4hr.py
python mapping-factor-estimation-30min.py
```

### RAS-MAPK SOS-RAF analysis

```bash
python SOS-RAF-analysis.py
```

---

## Author

**Mintu Nandi**  
Universal Biology Institute, The University of Tokyo  
7-3-1 Hongo, Bunkyo-ku, Tokyo 113-0033, Japan
