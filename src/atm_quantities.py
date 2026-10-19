"""
ATM implied-volatility, skew, and curvature computation.

Given model quantities (p_T, q_T, density-at-zero f_RT(0)), computes:

  ATM IV:    I(T) = (2/sqrt(T)) * Phi^{-1}((1 + q_T - p_T) / 2)

  ATM skew:  dI/dk = [Phi(-I*sqrt(T)/2) - p_T] / [sqrt(T) * phi(I*sqrt(T)/2)]

  ATM curvature:
    d²I/dk² = [d²C/dk² - d²C^BS/dk² - 2*(dC^BS/dkdsigma)*skew
               - (d²C^BS/dsigma²)*(skew²)] / vega

  where all BS quantities are evaluated at (T, k*=0, sigma=I(T)).

Analytical Black-Scholes ATM Greeks used (no numerical differentiation):
  d1 = I*sqrt(T)/2,  d2 = -I*sqrt(T)/2  (at k*=0, S0=1, r=0)
  vega          = sqrt(T) * phi(d1)
  d²C^BS/dk²   = phi(d2)/(I*sqrt(T)) - Phi(d2)       [at k=0]
  vanna BS      = phi(d1) * sqrt(T) / 2                [dC^BS/dk/dsigma at k=0]
  volga BS      = vega * d1 * d2 / I   [d²C^BS/dsigma² at k=0; = -I*T*sqrt(T)*phi(d1)/4]

# Custom — Context7 found no library equivalent for these exact
# rough-model ATM closed-form formulas; standard BS Greeks derived analytically.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def atm_iv(T: float, p_T: float, q_T: float) -> float:
    """
    Compute ATM implied volatility from Gil-Pelaez quantities.

    I(T) = (2 / sqrt(T)) * Phi^{-1}((1 + q_T - p_T) / 2)

    Parameters
    ----------
    T : float  maturity (> 0)
    p_T : float  P(X_T > 0)
    q_T : float  E[exp(X_T) 1_{X_T > 0}]

    Returns
    -------
    float  ATM implied volatility
    """
    assert T > 0, "Maturity must be positive."
    arg = (1.0 + q_T - p_T) / 2.0
    # Clamp to valid domain for Phi^{-1}
    arg = np.clip(arg, 1e-15, 1.0 - 1e-15)
    I = (2.0 / np.sqrt(T)) * norm.ppf(arg)
    return float(I)


def atm_skew(T: float, p_T: float, I: float) -> float:
    """
    Compute ATM implied-volatility skew (first log-strike derivative at k*=0).

    dI/dk|_{k=k*} = [Phi(-I*sqrt(T)/2) - p_T] / [sqrt(T) * phi(I*sqrt(T)/2)]

    Parameters
    ----------
    T : float  maturity
    p_T : float  P(X_T > 0)
    I : float  ATM implied volatility

    Returns
    -------
    float  ATM skew (negative for typical negatively skewed models)
    """
    sqrt_T = np.sqrt(T)
    half_I_sqrtT = I * sqrt_T / 2.0
    numerator = norm.cdf(-half_I_sqrtT) - p_T
    denominator = sqrt_T * norm.pdf(half_I_sqrtT)
    # Guard against denominator collapse at very small T (phi -> large but T->0)
    if abs(denominator) < 1e-300:
        return float("nan")
    return float(numerator / denominator)


def atm_curvature(
    T: float,
    p_T: float,
    q_T: float,
    I: float,
    skew: float,
    f0: float,
) -> float:
    """
    Compute ATM implied-volatility curvature (second log-strike derivative at k*=0).

    Uses the relationship between market call price and BS formula:
      C(T, k) = C^BS(T, k, I(T,k))

    Differentiating twice and solving for d²I/dk²:

      d²I/dk² = [d²C/dk²|model - d²C^BS/dk²|sigma=I
                 - 2 * vanna_BS * skew
                 - volga_BS * skew²] / vega_BS

    ATM call price:       C = q_T - p_T
    d²C/dk²|model at k=0: f_RT(0) - p_T   (Breeden-Litzenberger)
    d²C^BS/dk²  at k=0:  phi(d2)/(I*sqrt(T)) - Phi(d2)  =  phi*/(I*sqrtT) - Phi(-ISqrtT/2)
    vega_BS = sqrt(T) * phi(d1)
    vanna_BS = phi(d1) * sqrt(T) / 2
    volga_BS = T * d1 * d2 * phi(d1) / I  where d2 = -d1

    Parameters
    ----------
    T : float  maturity
    p_T : float
    q_T : float
    I : float  ATM implied volatility
    skew : float  ATM skew dI/dk
    f0 : float  density f_{RT}(0)

    Returns
    -------
    float  ATM curvature d²I/dk²
    """
    sqrt_T = np.sqrt(T)
    d1 = I * sqrt_T / 2.0     # d1 = d2 = ±I*sqrt(T)/2 at ATM (k=0)
    d2 = -d1
    phi_d1 = norm.pdf(d1)
    Phi_d2 = norm.cdf(d2)     # = Phi(-I*sqrt(T)/2)

    # d²C/dk² from model (Breeden-Litzenberger at k*=0)
    d2C_model = f0 - p_T

    # d²C^BS/dk² at ATM (k=0)
    # From d/dk[-e^k Phi(d2)] = -e^k Phi(d2) + e^k phi(d2)/sigma/sqrt(T)  (simplified at ATM)
    # At k=0: phi(d2)/(I*sqrt(T)) - Phi(d2)
    d2C_BS = phi_d1 / (I * sqrt_T) - Phi_d2  # phi_d1 = phi_d2 by symmetry at ATM

    # vega  = sqrt(T) * phi(d1)
    vega = sqrt_T * phi_d1

    # vanna = dC^BS/(dk dsigma) at ATM (k=0)
    # = phi(d1) * sqrt(T) / 2  (see derivation in module docstring)
    vanna = phi_d1 * sqrt_T / 2.0

    # volga = d²C^BS/dsigma² at ATM (k=0)
    # Standard result: volga = vega * d1 * d2 / sigma
    # vega = sqrt(T) * phi(d1), d1=I*sqrt(T)/2, d2=-I*sqrt(T)/2
    # => volga = sqrt(T)*phi(d1) * (I*sqrt(T)/2)*(-I*sqrt(T)/2) / I
    #          = phi(d1) * (-I * T * sqrt(T) / 4)
    volga = vega * d1 * d2 / I if abs(I) > 1e-12 else 0.0

    if abs(vega) < 1e-300:
        return float("nan")

    curvature = (d2C_model - d2C_BS - 2.0 * vanna * skew - volga * skew**2) / vega
    return float(curvature)


def local_log_slope(
    log_T: np.ndarray,
    log_abs_qty: np.ndarray,
) -> np.ndarray:
    """
    Estimate local log-log slope (power-law exponent) via consecutive differences.

    hat_beta(T_i) = [log|qty(T_{i+1})| - log|qty(T_i)|] / [log T_{i+1} - log T_i]

    Parameters
    ----------
    log_T : ndarray, shape (N,)  log of maturity grid
    log_abs_qty : ndarray, shape (N,)  log of |qty| (skew or curvature)

    Returns
    -------
    ndarray, shape (N-1,)  local slope estimates (midpoint indexed)
    """
    slopes = np.diff(log_abs_qty) / np.diff(log_T)
    return slopes


def theoretical_skew_exponent(H: float, Y: float) -> float:
    """
    Theoretical ATM skew power-law exponent.

    beta_s = min(H - 1/2, (1 - Y) / 2)

    Parameters
    ----------
    H : float  Hurst exponent
    Y : float  CGMY stability index

    Returns
    -------
    float
    """
    return min(H - 0.5, (1.0 - Y) / 2.0)


def theoretical_curvature_exponent(H: float, Y: float) -> float:
    """
    Theoretical ATM curvature power-law exponent.

    beta_c = min(2H - 1, -Y/2)

    Parameters
    ----------
    H : float  Hurst exponent
    Y : float  CGMY stability index

    Returns
    -------
    float
    """
    return min(2.0 * H - 1.0, -Y / 2.0)


def rough_dominated_skew_coefficient(
    V0: float, vol_of_vol: float, rho: float, H: float
) -> float:
    """
    Theoretical leading coefficient A_s in rough-dominated skew regime.

    A_s = rho * nu / [2 * sqrt(V0) * (H + 1/2) * (H + 3/2) * Gamma(H + 1/2)]

    Parameters
    ----------
    V0 : float
    vol_of_vol : float  (nu)
    rho : float
    H : float

    Returns
    -------
    float
    """
    from scipy.special import gamma as sc_gamma
    nu = vol_of_vol
    denom = (
        2.0 * np.sqrt(V0)
        * (H + 0.5)
        * (H + 1.5)
        * sc_gamma(H + 0.5)
    )
    return float(rho * nu / denom)


def jump_dominated_skew_coefficient(
    V0: float,
    C_plus: float,
    C_minus: float,
    Y: float,
) -> float:
    """
    Theoretical leading coefficient A_s in jump-dominated skew regime.

    A_s = (C_plus - C_minus) * Gamma((3 - Y)/2)
          / [V0^{Y/2} * Y * (Y - 1) * 2^{(Y-1)/2}]

    Parameters
    ----------
    V0 : float
    C_plus, C_minus : float  CGMY asymmetry parameters
    Y : float

    Returns
    -------
    float
    """
    from scipy.special import gamma as sc_gamma
    num = (C_plus - C_minus) * sc_gamma((3.0 - Y) / 2.0)
    denom = (V0 ** (Y / 2.0)) * Y * (Y - 1.0) * (2.0 ** ((Y - 1.0) / 2.0))
    return float(num / denom)
