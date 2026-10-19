"""
Rough Heston characteristic function via fractional Riccati equation.

Implements the rough Heston model of El Euch & Rosenbaum (2019):
  V_t = V0 + kappa * int_0^t K(t-s)(theta - V_s) ds
            + vol_of_vol * int_0^t K(t-s) sqrt(V_s) dW_s
  K(t) = t^{H - 1/2} / Gamma(H + 1/2)

The characteristic function of the log-return RT = X_T - X_0 is:
  phi_T^cont(z) = exp(V0 * I^{1-alpha} h_z(T) + kappa*theta * int_0^T h_z(s) ds)

where alpha = H + 1/2, h_z satisfies the Volterra integral equation:
  h_z(t) = I^alpha F(z, h_z)(t)
  F(z, x) = -1/2 (z^2 + iz) + (rho*nu*iz - kappa)*x + (nu^2/2)*x^2

Solved numerically via the fractional Adams-Bashforth (rectangle) scheme:
  h(t_n) = sum_{j=0}^{n-1} w[n-j] * F(z, h(t_j))
  w[m] = dt^alpha / Gamma(alpha+1) * [m^alpha - (m-1)^alpha]

# Custom — Context7 found no library equivalent for rough Heston
# fractional Riccati solver with PECE/Euler scheme (paper methodology).
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma


def _F(
    z: np.ndarray,
    x: np.ndarray,
    kappa: float,
    theta: float,
    vol_of_vol: float,
    rho: float,
) -> np.ndarray:
    """
    Riccati kernel function.

    F(z, x) = -1/2*(z^2 + iz) + (rho*nu*iz - kappa)*x + (nu^2/2)*x^2

    Parameters
    ----------
    z : ndarray, shape (N_z,)
    x : ndarray, shape (N_z,)  current solution value
    kappa, theta, vol_of_vol, rho : float  rough Heston parameters

    Returns
    -------
    ndarray, shape (N_z,)
    """
    nu = vol_of_vol
    return (
        -0.5 * (z**2 + 1j * z)
        + (rho * nu * 1j * z - kappa) * x
        + (nu**2 / 2.0) * x**2
    )


def solve_fractional_riccati(
    z_array: np.ndarray,
    T: float,
    V0: float,
    vol_of_vol: float,
    kappa: float,
    theta: float,
    rho: float,
    H: float,
    N_steps: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the fractional Volterra Riccati equation for an array of z values.

    Uses the explicit rectangle (Adams-Bashforth, order-1) scheme for:
      h_z(t_n) = (1/Gamma(alpha)) * sum_{j<n} F(z, h_z(t_j)) *
                 int_{t_j}^{t_{j+1}} (t_n - s)^{alpha-1} ds

    Also computes:
      I^{1-alpha} h_z(T) = (1/Gamma(1-alpha)) * int_0^T (T-s)^{-alpha} h_z(s) ds
      int_0^T h_z(s) ds

    Parameters
    ----------
    z_array : ndarray, shape (N_z,)
        Array of Fourier arguments (complex).
    T : float
        Maturity (must be > 0).
    V0 : float
        Initial variance.
    vol_of_vol : float
        Volatility-of-volatility (nu in the paper).
    kappa : float
        Mean-reversion speed.
    theta : float
        Long-run variance.
    rho : float
        Correlation between price and variance Brownian motions.
    H : float
        Hurst exponent (0 < H < 1/2 for rough volatility).
    N_steps : int
        Number of time steps for fractional Euler discretisation.

    Returns
    -------
    h_T : ndarray, shape (N_z,)
        h_z(T) — terminal value of the Riccati solution.
    I1ma_h_T : ndarray, shape (N_z,)
        I^{1-alpha} h_z(T) — Riemann-Liouville fractional integral of h_z.
    int_h : ndarray, shape (N_z,)
        int_0^T h_z(s) ds — ordinary integral of h_z over [0, T].
    """
    alpha = H + 0.5  # alpha in (0.5, 1.0) since H in (0, 0.5)
    dt = T / N_steps
    N = N_steps
    N_z = len(z_array)

    # Precompute weights for the fractional rectangle rule
    # w[m] = dt^alpha / Gamma(alpha+1) * [m^alpha - (m-1)^alpha], m = 1, ..., N
    # Shape: (N,) — index m corresponds to lag m in the convolution
    m = np.arange(1, N + 1, dtype=np.float64)
    # Using scipy.special.gamma — Context7 confirmed
    w_riccati = (dt**alpha / gamma(alpha + 1.0)) * (m**alpha - (m - 1.0) ** alpha)

    # Storage: h_steps[n] = h_z(t_n), shape (N_z,), h_steps[0] = 0
    h_steps = np.zeros((N + 1, N_z), dtype=complex)

    # F values: F_steps[n] = F(z, h_z(t_n)), computed after each h step
    F_steps = np.zeros((N + 1, N_z), dtype=complex)
    F_steps[0] = _F(z_array, h_steps[0], kappa, theta, vol_of_vol, rho)

    # March forward: h(t_n) = sum_{j=0}^{n-1} w[n-j] * F(z, h(t_j))
    for n in range(1, N + 1):
        # Convolution: sum over j = 0, ..., n-1; lag = n-j = n, n-1, ..., 1
        lags = np.arange(n, 0, -1)          # shape (n,): lags n, n-1, ..., 1
        wts = w_riccati[lags - 1]            # w[lag], shape (n,)
        # h[n] = sum_j w[n-j] * F(z, h[j])
        # F_steps[0:n] shape (n, N_z); wts shape (n,)
        h_steps[n] = np.dot(wts, F_steps[:n])
        # Safety: if h overflows, the CF is negligibly small; cap h
        bad = ~np.isfinite(h_steps[n])
        if np.any(bad):
            h_steps[n][bad] = complex(-1e10, 0)
        if n < N:
            F_steps[n] = _F(z_array, h_steps[n], kappa, theta, vol_of_vol, rho)
            # Zero out F at capped locations to prevent error propagation
            if np.any(bad):
                F_steps[n][bad] = 0j

    # Compute I^{1-alpha} h_z(T) using rectangle rule
    # I^{1-alpha} h_z(T) = (1/Gamma(1-alpha)) * int_0^T (T-s)^{-alpha} h_z(s) ds
    # Approximate using h_z(s) constant on [t_j, t_{j+1}] = h_steps[j]
    # int_{t_j}^{t_{j+1}} (T-s)^{-alpha} ds
    #   = [(T-t_j)^{1-alpha} - (T-t_{j+1})^{1-alpha}] / (1-alpha)
    #   = dt^{1-alpha} [(N-j)^{1-alpha} - (N-j-1)^{1-alpha}] / (1-alpha)
    j_idx = np.arange(0, N, dtype=np.float64)
    w_I1ma = (dt ** (1.0 - alpha) / gamma(2.0 - alpha)) * (
        (N - j_idx) ** (1.0 - alpha) - (N - j_idx - 1.0) ** (1.0 - alpha)
    )  # shape (N,)
    # I^{1-alpha} h_z(T) = sum_{j=0}^{N-1} w_I1ma[j] * h_steps[j]
    I1ma_h_T = np.dot(w_I1ma, h_steps[:N])   # shape (N_z,)

    # Compute int_0^T h_z(s) ds via rectangle rule (using h_steps[j] for interval j)
    int_h = dt * np.sum(h_steps[:N], axis=0)  # shape (N_z,)

    h_T = h_steps[N]
    return h_T, I1ma_h_T, int_h


def rough_heston_cf(
    z_array: np.ndarray,
    T: float,
    V0: float,
    vol_of_vol: float,
    kappa: float,
    theta: float,
    rho: float,
    H: float,
    N_steps: int = 200,
) -> np.ndarray:
    """
    Compute the rough Heston (continuous part) characteristic function.

    phi_T^cont(z) = exp(V0 * I^{1-alpha} h_z(T) + kappa*theta * int_0^T h_z(s) ds)

    Parameters
    ----------
    z_array : ndarray, shape (N_z,)
    T : float
    V0, vol_of_vol, kappa, theta, rho, H : float  rough Heston parameters
    N_steps : int

    Returns
    -------
    ndarray, shape (N_z,) of complex
    """
    _, I1ma_h_T, int_h = solve_fractional_riccati(
        z_array, T, V0, vol_of_vol, kappa, theta, rho, H, N_steps
    )
    log_cf = V0 * I1ma_h_T + kappa * theta * int_h
    return np.exp(log_cf)


def rough_heston_cf_pure(
    z_array: np.ndarray,
    T: float,
    V0: float,
    vol_of_vol: float,
    kappa: float,
    theta: float,
    rho: float,
    H: float,
    N_steps: int = 200,
) -> np.ndarray:
    """
    Alias for rough_heston_cf (rough-only model, no jumps).

    Parameters
    ----------
    Same as rough_heston_cf.

    Returns
    -------
    ndarray, shape (N_z,) of complex
    """
    return rough_heston_cf(z_array, T, V0, vol_of_vol, kappa, theta, rho, H, N_steps)
