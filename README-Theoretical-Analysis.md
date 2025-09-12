# Theoretical Analysis in Mathematica

This directory contains Mathematica (`.nb`) notebooks with symbolic derivations and analytical exploration of **mutual information (MI)**, **2-Wasserstein distance (WD)** due to varying BAP regimes and the Lagrange multiplier in canonical gene regulatory motifs.

These notebooks provide the **theoretical foundation** and enable motif-specific symbolic results and steady-state calculations of MI and 2-WD under Gaussian approximation.

---

## Contents

### Core Motifs
- **`SC.nb`** – Single Cascade.
- **`C1-FFL.nb`** – Coherent type-I Feed-Forward Loop.
- **`I1-FFL.nb`** – Incoherent type-I Feed-Forward Loop.
- **`PFL.nb`** – Positive Feedback Loop
- **`DNFL.nb`** – Double Negative Feedback Loop.
- **`NFL.nb`** – Negative Feedback Loop.

### Single Cascade (SC)
One notebook covering all three BAP regimes: Theta_X > 1, Theta_X = 1, and Theta_X < 1

### Coherent type-I Feed-Forward Loop (C1-FFL)
One notebook covering all three BAP regimes: Theta_X > 1, Theta_X = 1, and Theta_X < 1

### Incoherent type-I Feed-Forward Loop (I1-FFL)
One notebook covering all three BAP regimes: Theta_X > 1, Theta_X = 1, and Theta_X < 1

### Positive Feedback Loop (PFL)
Nine notebooks covering all BAP regime combinations (Theta_X, Theta_Z) for PFL:
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
Nine notebooks covering all BAP regime combinations (Theta_X, Theta_Z) for DNFL:
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
Nine notebooks covering all BAP regime combinations (Theta_X, Theta_Z) for NFL:
- `NFL-eq-1-eq-1.nb`
- `NFL-eq-1-gr-1.nb`
- `NFL-eq-1-less-1.nb`
- `NFL-gr-1-eq-1.nb`
- `NFL-gr-1-gr-1.nb`
- `NFL-gr-1-less-1.nb`
- `NFL-less-1-eq-1.nb`
- `NFL-less-1-gr-1.nb`
- `NFL-less-1-less-1.nb`

---

## Usage

1. Open any notebook in **Wolfram Mathematica** (any version).
2. Evaluate cells sequentially to reproduce symbolic derivations.
3. Each notebook corresponds to a specific motif and BAP regime.
4. Outputs include:
   - Numerical values of MI and 2-WD with proper export as .dat file.

---
---

## Author

**Mintu Nandi**  
Universal Biology Institute, The University of Tokyo, 7-3-1 Hongo, Bunkyo-ku, Tokyo, Japan
