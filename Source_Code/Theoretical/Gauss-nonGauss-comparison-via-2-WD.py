"""
Non-Gaussian relay example comparing MI and 2-Wasserstein distance.

Correspondence with the manuscript
----------------------------------
This script provides an illustrative non-Gaussian example used to clarify the
added value of the 2-Wasserstein distance beyond mutual information. The same
unimodal input distribution is transmitted through two different relays:

    1. A graded relay that generates a unimodal output.
    2. A switch-like heterogeneous relay that generates a bimodal output.

The graded relay noise is tuned so that the graded and switch-like relays have
comparable mutual information. The resulting 2-Wasserstein distances are then
compared to show that MI captures input-output statistical dependence, whereas
2-WD captures geometric reorganization of the response distribution.

Main calculations
-----------------
1. Define a positive unimodal input distribution X using a Gamma distribution.
2. Construct conditional output distributions for a graded relay and a
   switch-like relay.
3. Compute MI from the full joint distribution p(x, z).
4. Compute the 1D 2-Wasserstein distance from input and output quantile
   functions.
5. Export probability-density profiles and MI/2-WD summary values for
   plotting in OriginPro.

Output files
------------
    nonGaussian_relay_probability_profiles.dat
    nonGaussian_relay_MI_WD_summary.dat

Author
------
Mintu Nandi
Universal Biology Institute, The University of Tokyo
"""






import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import gamma, norm
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import brentq


# ============================================================
# 1. Numerical grids
# ============================================================

# Input X is positive and concentration-like.
# X will be concentrated around mean 3 with SD 0.5.
x_grid = np.linspace(1e-6, 6.0, 3000)

# Use a wide output grid. The model is not truncated, but the
# distribution is shifted far enough to the right that negative
# output probability is negligible.
z_grid = np.linspace(-1.0, 7.0, 4000)

# Quantile grid for 1D 2-Wasserstein distance.
u_grid = np.linspace(1e-5, 1.0 - 1e-5, 30000)


# ============================================================
# 2. Input distribution
# ============================================================

# X ~ Gamma(alpha=36, beta=1/12)
# alpha = shape parameter
# beta  = scale parameter
#
# Mean = alpha * beta = 3
# SD   = sqrt(alpha) * beta = 0.5

alpha = 36.0
beta = 1.0 / 12.0

px = gamma.pdf(x_grid, a=alpha, scale=beta)
px = px / trapezoid(px, x_grid)

mean_x = trapezoid(x_grid * px, x_grid)
var_x = trapezoid((x_grid - mean_x)**2 * px, x_grid)
std_x = np.sqrt(var_x)

print("Input distribution")
print("Mean(X) =", mean_x)
print("SD(X)   =", std_x)


# ============================================================
# 3. Utility functions
# ============================================================

def normalize_density(grid, density):
    area = trapezoid(density, grid)
    return density / area


def inverse_cdf_from_density(grid, density):
    """
    Construct inverse CDF F^{-1}(u) from a 1D density.
    """
    density = normalize_density(grid, density)

    cdf = np.concatenate([
        [0.0],
        np.cumsum((density[:-1] + density[1:]) * np.diff(grid) / 2.0)
    ])
    cdf = cdf / cdf[-1]

    # Remove duplicate CDF values for stable interpolation.
    cdf_unique, idx = np.unique(cdf, return_index=True)
    grid_unique = grid[idx]

    inv_cdf = interp1d(
        cdf_unique,
        grid_unique,
        bounds_error=False,
        fill_value=(grid_unique[0], grid_unique[-1])
    )

    return inv_cdf


def compute_mi_and_w2(px, x_grid, pz_given_x, z_grid, u_grid):
    """
    Compute MI and 2-Wasserstein distance from full distributions.

    Joint distribution:
        p(x,z) = p_X(x) p(z|x)

    Output marginal:
        p_Z(z) = int p_X(x) p(z|x) dx

    Mutual information:
        I(X;Z) = int dx dz p_X(x)p(z|x)
                 log2[p(z|x)/p_Z(z)]

    1D 2-Wasserstein distance:
        W^2 = int_0^1 [F_X^{-1}(u)-F_Z^{-1}(u)]^2 du
    """

    # Normalize each conditional density p(z|x) over the finite z-grid.
    row_norms = trapezoid(pz_given_x, z_grid, axis=1)
    pz_given_x = pz_given_x / row_norms[:, None]

    # Output marginal p_Z(z)
    pz = trapezoid(px[:, None] * pz_given_x, x_grid, axis=0)
    pz = normalize_density(z_grid, pz)

    # Mutual information
    eps = 1e-300
    ratio = np.maximum(pz_given_x, eps) / np.maximum(pz[None, :], eps)

    mi_integrand = px[:, None] * pz_given_x * np.log2(ratio)
    mi_z_integrated = trapezoid(mi_integrand, z_grid, axis=1)
    mi = trapezoid(mi_z_integrated, x_grid)

    # 2-Wasserstein distance using the quantile formula
    inv_fx = inverse_cdf_from_density(x_grid, px)
    inv_fz = inverse_cdf_from_density(z_grid, pz)

    w2_squared = trapezoid(
        (inv_fx(u_grid) - inv_fz(u_grid))**2,
        u_grid
    )
    w2 = np.sqrt(w2_squared)

    return mi, w2, pz


# ============================================================
# 4. Graded relay model
# ============================================================

def graded_relay_pz_given_x(gamma_g):
    """
    Graded relay model:

        Z_G = X + Gamma_G * xi_G

    where xi_G ~ N(0,1).

    Therefore:

        P(Z_G=z | X=x) = N(z; x, Gamma_G^2)

    Gamma_G is used only in the graded relay.
    """

    pz_given_x = norm.pdf(
        z_grid[None, :],
        loc=x_grid[:, None],
        scale=gamma_g
    )

    return pz_given_x


# ============================================================
# 5. Switch-like heterogeneous relay model
# ============================================================

def switch_relay_pz_given_x(
    hill_h=16.0,
    K=3.0,
    z_low=1.5,
    z_high=4.5,
    sigma_b=0.15
):
    """
    Switch-like heterogeneous relay.

    Introduce a latent activation state S:

        S = 0 : low-response state
        S = 1 : high-response state

    The upstream signal controls the probability of high activation:

        P(S=1 | X=x) = s(x) = x^h / (K^h + x^h)

    The output is then sampled from one of two Gaussian response states:

        Z_B = z_low  + sigma_B * xi_B, if S=0
        Z_B = z_high + sigma_B * xi_B, if S=1

    where xi_B ~ N(0,1).

    Therefore:

        P(Z_B=z | X=x)
        = [1-s(x)] N(z; z_low, sigma_B^2)
          + s(x) N(z; z_high, sigma_B^2)

    Gamma_G is not used here. The switch-like relay has its own
    response noise sigma_B.
    """

    s = x_grid**hill_h / (K**hill_h + x_grid**hill_h)

    low_density = norm.pdf(
        z_grid[None, :],
        loc=z_low,
        scale=sigma_b
    )

    high_density = norm.pdf(
        z_grid[None, :],
        loc=z_high,
        scale=sigma_b
    )

    pz_given_x = (
        (1.0 - s)[:, None] * low_density
        + s[:, None] * high_density
    )

    return pz_given_x


# ============================================================
# 6. Compute switch-like relay first
# ============================================================

pz_given_x_switch = switch_relay_pz_given_x(
    hill_h=16.0,
    K=3.0,
    z_low=1.5,
    z_high=4.5,
    sigma_b=0.15
)

mi_switch, w2_switch, pz_switch = compute_mi_and_w2(
    px=px,
    x_grid=x_grid,
    pz_given_x=pz_given_x_switch,
    z_grid=z_grid,
    u_grid=u_grid
)


# ============================================================
# 7. Tune Gamma_G so graded relay has matched MI
# ============================================================

def graded_mi_minus_target(gamma_g):
    pz_given_x_graded = graded_relay_pz_given_x(gamma_g)

    mi_graded, _, _ = compute_mi_and_w2(
        px=px,
        x_grid=x_grid,
        pz_given_x=pz_given_x_graded,
        z_grid=z_grid,
        u_grid=u_grid
    )

    return mi_graded - mi_switch


gamma_g_matched = brentq(
    graded_mi_minus_target,
    0.05,
    3.0
)

pz_given_x_graded = graded_relay_pz_given_x(gamma_g_matched)

mi_graded, w2_graded, pz_graded = compute_mi_and_w2(
    px=px,
    x_grid=x_grid,
    pz_given_x=pz_given_x_graded,
    z_grid=z_grid,
    u_grid=u_grid
)


# ============================================================
# 8. Check negative probabilities
# ============================================================

# For the graded relay, Z_G = X + Gamma_G * xi.
# Approximate marginal variance is Var(X) + Gamma_G^2.
# This approximate probability should be very small.
approx_p_negative_graded = norm.cdf(
    0.0,
    loc=mean_x,
    scale=np.sqrt(std_x**2 + gamma_g_matched**2)
)

p_negative_low = norm.cdf(
    0.0,
    loc=1.5,
    scale=0.15
)

p_negative_high = norm.cdf(
    0.0,
    loc=4.5,
    scale=0.15
)


# ============================================================
# 9. Print final results
# ============================================================

print("\nMatched graded noise")
print("Gamma_G =", gamma_g_matched)

print("\nGraded relay")
print("MI I(X;Z_G) =", mi_graded, "bits")
print("W(X,Z_G)    =", w2_graded)
print("W^-1        =", 1.0 / w2_graded)

print("\nSwitch-like relay")
print("MI I(X;Z_B) =", mi_switch, "bits")
print("W(X,Z_B)    =", w2_switch)
print("W^-1        =", 1.0 / w2_switch)

print("\nApproximate probability of negative response")
print("P(Z_G < 0) approximately =", approx_p_negative_graded)
print("P(low-state Z_B < 0)     =", p_negative_low)
print("P(high-state Z_B < 0)    =", p_negative_high)


# ============================================================
# 10. Plot input and output marginal distributions
# ============================================================

plt.figure(figsize=(7, 4.5))

plt.plot(x_grid, px, label="Input X")
plt.plot(z_grid, pz_graded, label="Graded output Z_G")
plt.plot(z_grid, pz_switch, label="Switch-like output Z_B")

plt.xlabel("Normalized activity")
plt.ylabel("Probability density")
plt.xlim(0, 6)
plt.legend()
plt.tight_layout()
plt.show()



# ============================================================
# 11. Export probability-density data for Origin Pro
# ============================================================

# Common grid for plotting all three distributions together.
# This corresponds to the x-axis of the figure: normalized signaling variable.
activity_grid = np.linspace(0.0, 6.0, 4000)

# Interpolate input and output marginal densities onto the common grid.
px_interp = np.interp(
    activity_grid,
    x_grid,
    px,
    left=0.0,
    right=0.0
)

pz_graded_interp = np.interp(
    activity_grid,
    z_grid,
    pz_graded,
    left=0.0,
    right=0.0
)

pz_switch_interp = np.interp(
    activity_grid,
    z_grid,
    pz_switch,
    left=0.0,
    right=0.0
)

# Combine into one Origin-compatible table.
# Columns:
# 1. Normalized signaling variable
# 2. Input probability density P_X
# 3. Graded output probability density P_ZG
# 4. Switch-like output probability density P_ZB
probability_data = np.column_stack([
    activity_grid,
    px_interp,
    pz_graded_interp,
    pz_switch_interp
])

np.savetxt(
    "nonGaussian_relay_probability_profiles.dat",
    probability_data,
    fmt="%.10e",
    delimiter="\t",
    header=(
        "Normalized_signaling_variable\t"
        "P_X_input\t"
        "P_ZG_graded_output\t"
        "P_ZB_switch_like_output"
    ),
    comments=""
)


# ============================================================
# 12. Export MI and 2-WD summary values for Origin Pro
# ============================================================

summary_data = np.array([
    [1, mi_graded, w2_graded, 1.0 / w2_graded, gamma_g_matched],
    [2, mi_switch, w2_switch, 1.0 / w2_switch, np.nan]
])

np.savetxt(
    "nonGaussian_relay_MI_WD_summary.dat",
    summary_data,
    fmt=["%d", "%.10e", "%.10e", "%.10e", "%.10e"],
    delimiter="\t",
    header=(
        "Relay_ID\t"
        "MI_bits\t"
        "W2_distance\t"
        "Geometric_fidelity_W2_inverse\t"
        "Gamma_G"
    ),
    comments=""
)

print("\nExported files:")
print("1. nonGaussian_relay_probability_profiles.dat")
print("2. nonGaussian_relay_MI_WD_summary.dat")