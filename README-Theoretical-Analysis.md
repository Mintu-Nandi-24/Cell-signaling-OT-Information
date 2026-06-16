# Theoretical Analysis in Mathematica

This directory contains Mathematica (`.nb`) notebooks with symbolic derivations and analytical exploration of **mutual information (MI)**, **2-Wasserstein distance (2-WD)**, and biochemical allocation parameter (BAP) regimes in canonical gene regulatory motifs.

These notebooks provide the **theoretical foundation** for the dual-fidelity framework by deriving motif-specific noise, covariance, MI, and 2-WD expressions under coarse-grained stochastic descriptions.

---

## Contents

### Core Motifs
- **`SC.nb`** – Single Cascade (SC).
- **`C1-FFL.nb`** – Coherent Feed-Forward Loop.
- **`I1-FFL.nb`** – Incoherent Feed-Forward Loop.

### Cooperative-Binding Check
- **`C1-FFL-h2.nb`** – Representative weak-cooperativity check for C1-FFL with Hill coefficient `h = 2`. This notebook tests whether weak cooperativity preserves the qualitative dual-fidelity behavior observed in the non-cooperative model.

### Positive Feedback Loop (PFL)
Nine notebooks covering all BAP regime combinations for PFL:
- `PFL-eq-1-eq-1.nb`
- `PFL-eq-1-gr-1.nb`
- `PFL-eq-1-less-1.nb`
- `PFL-gr-1-eq-1.nb`
- `PFL-gr-1-gr-1.nb`
- `PFL-gr-1-less-1.nb`
- `PFL-less-1-eq-1.nb`
- `PFL-less-1-gr-1.nb`
- `PFL-less-1-less-1.nb`

### Dual Negative Feedback Loop (DNFL)
Nine notebooks covering all BAP regime combinations for DNFL:
- `DNFL-eq-1-eq-1.nb`
- `DNFL-eq-1-gr-1.nb`
- `DNFL-eq-1-less-1.nb`
- `DNFL-gr-1-eq-1.nb`
- `DNFL-gr-1-gr-1.nb`
- `DNFL-gr-1-less-1.nb`
- `DNFL-less-1-eq-1.nb`
- `DNFL-less-1-gr-1.nb`
- `DNFL-less-1-less-1.nb`

### Negative Feedback Loop (NFL)
Nine notebooks covering all BAP regime combinations for NFL:
- `NFL-eq-1-eq-1.nb`
- `NFL-eq-1-gr-1.nb`
- `NFL-eq-1-less-1.nb`
- `NFL-gr-1-eq-1.nb`
- `NFL-gr-1-gr-1.nb`
- `NFL-gr-1-less-1.nb`
- `NFL-less-1-eq-1.nb`
- `NFL-less-1-gr-1.nb`
- `NFL-less-1-less-1.nb`

### Non-Gaussian Illustrative Example
- **`Gauss-nonGauss-comparison-via-2-WD.py`** – Python script showing how 2-WD distinguishes a graded unimodal relay from a switch-like bimodal relay even when MI is matched between the two relays.

---

## Usage

### Mathematica notebooks
1. Open any notebook in **Wolfram Mathematica** (v12 or later recommended).
2. Evaluate cells sequentially to reproduce symbolic derivations and motif-specific calculations.
3. Each notebook corresponds to a specific motif or BAP regime.

Outputs include:
- Symbolic noise, covariance, MI, and 2-WD expressions.
- Regime-specific simplifications.
- Steady-state and fidelity analyses.
- Exported numerical values for plotting when applicable.

### Python non-Gaussian example
Install dependencies:

```bash
pip install numpy scipy matplotlib
```

Run:

```bash
python Gauss-nonGauss-comparison-via-2-WD.py
```

This produces:
- `nonGaussian_relay_probability_profiles.dat`
- `nonGaussian_relay_MI_WD_summary.dat`

---

## Notes

The motif notebooks use coarse-grained Hill-function and Langevin-based descriptions. They are intended to reveal qualitative principles governing the balance between informational fidelity and geometric fidelity, rather than to provide complete molecular promoter models.

---

## Author

**Mintu Nandi**  
Universal Biology Institute, The University of Tokyo
