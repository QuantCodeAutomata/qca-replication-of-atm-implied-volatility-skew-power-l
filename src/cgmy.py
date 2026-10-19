"""
CGMY characteristic exponent module.

Implements the CGMY (Carr-Geman-Madan-Yor) Lévy process cumulant
as required by the mixed rough Heston + CGMY model.

# Custom — Context7 found no library equivalent; direct implementation
# of paper formula for chi(z) = Gamma(-Y)*[C+*((M-iz)^Y - M^Y) + C-*((G+iz)^Y - G^Y)]
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma


def cgmy_exponent(
    z: complex | np.ndarray,
    C_plus: float,
    C_minus: float,
    G: float,
    M: float,
    Y: float,
) -> complex | np.ndarray:
    """
    Compute the CGMY Lévy cumulant chi(z).

    Defined as:
        chi(z) = Gamma(-Y) * [C_plus * ((M - i*z)^Y - M^Y)
                             + C_minus * ((G + i*z)^Y - G^Y)]

    Uses the principal branch for complex powers throughout.

    Parameters
    ----------
    z : complex or array of complex
        Fourier argument (can be real or complex).
    C_plus : float
        Positive-jump intensity parameter (C_plus > 0).
    C_minus : float
        Negative-jump intensity parameter (C_minus > 0).
    G : float
        Decay rate on the negative side (G > 0).
    M : float
        Decay rate on the positive side (M > 0, M > 1 required for finite
        first moment and martingale condition).
    Y : float
        Stability index (Y in (1, 2) for infinite variation).

    Returns
    -------
    complex or array of complex
        The CGMY cumulant chi(z).

    Notes
    -----
    Principal branch for complex power: w^a = exp(a * log(w)) where
    log is the principal branch (imaginary part in (-pi, pi]).
    For M - i*z with z = u > 0: arg(M - iu) in (-pi/2, 0) -- safe.
    For G + i*z with z = u > 0: arg(G + iu) in (0, pi/2) -- safe.
    """
    # Gamma(-Y) using scipy.special.gamma; valid for Y not a non-negative integer
    # Using scipy.special.gamma — Context7 confirmed for scipy
    gam_neg_Y = gamma(-Y)

    # Principal-branch complex powers
    term_plus = C_plus * (_cpow(M - 1j * z, Y) - M**Y)
    term_minus = C_minus * (_cpow(G + 1j * z, Y) - G**Y)

    return gam_neg_Y * (term_plus + term_minus)


def cgmy_drift_compensation(
    C_plus: float,
    C_minus: float,
    G: float,
    M: float,
    Y: float,
) -> complex:
    """
    Compute chi(-i), the log-MGF of the CGMY jump at unit time.

    This equals log E[exp(L_1)] where L is the CGMY process.
    Used to enforce the martingale condition:
        exp(X_t) is a martingale iff E[exp(L_t)] = exp(eta * t)
    where eta = chi(-i).

    Parameters
    ----------
    C_plus, C_minus, G, M, Y : float
        CGMY parameters (see cgmy_exponent docstring).

    Returns
    -------
    float
        chi(-i) = Gamma(-Y)*[C_plus*((M-1)^Y - M^Y) + C_minus*((G+1)^Y - G^Y)]
    """
    return float(np.real(cgmy_exponent(-1j, C_plus, C_minus, G, M, Y)))


def _cpow(base: complex | np.ndarray, exponent: float) -> complex | np.ndarray:
    """
    Complex power using the principal branch of the complex logarithm.

    w^a = exp(a * log(w)), log = principal branch.

    Parameters
    ----------
    base : complex or array of complex
    exponent : float

    Returns
    -------
    complex or array of complex
    """
    return np.exp(exponent * np.log(base.astype(complex) if isinstance(base, np.ndarray) else complex(base)))
