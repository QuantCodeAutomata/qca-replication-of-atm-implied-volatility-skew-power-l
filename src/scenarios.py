"""
Scenario configurations for rough Heston + CGMY ATM IV skew experiments.

Three skew scenarios from the paper (Exp 1 & 2):
  S1  rough-dominated skew:  beta_s = min(H-0.5, (1-Y)/2) = min(-0.4, -0.25) = -0.4
  S2  jump-dominated skew:   beta_s = min(H-0.5, (1-Y)/2) = min(-0.15, -0.25) = -0.25
  S3  critical skew line:    H = 1 - Y/2 => H = 0.25, beta_s = -0.25

Three skew-curvature scenarios (Exp 3):
  C1  rough-only model (L=0):             beta_s=-0.4, beta_c=-0.8
  C2  pure jump + constant vol (sigma=0.2): beta_s=-0.25, beta_c=-0.75
  C3  mixed rough Heston + CGMY:          beta_s=-0.3, beta_c=-0.75
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.fourier_inversion import RoughHestonParams, CGMYParams


@dataclass
class ScenarioConfig:
    """Full scenario configuration."""
    name: str
    label: str
    description: str
    rh: Optional[RoughHestonParams]      # None => no rough Heston (pure jump or const vol)
    cgmy: Optional[CGMYParams]           # None => no jumps
    sigma_const: Optional[float]         # constant vol (Exp 3, C2 only)
    beta_s_theory: float                 # theoretical skew exponent
    beta_c_theory: float                 # theoretical curvature exponent
    A_s_theory: Optional[float] = None   # theoretical leading skew coefficient


# ---------- Common CGMY parameters (all three skew scenarios) ----------
_CGMY_COMMON = CGMYParams(C_plus=0.015, C_minus=0.025, G=5.0, M=8.0, Y=1.5)


# ---------- Exp 1 & 2 scenarios ----------
SCENARIO_S1 = ScenarioConfig(
    name="S1",
    label="Rough-dominated skew",
    description="H=0.1, Y=1.5 => beta_s=min(-0.4,-0.25)=-0.4 (rough dominates)",
    rh=RoughHestonParams(V0=0.04, vol_of_vol=0.45, kappa=1.5, theta=0.04, rho=-0.7, H=0.1),
    cgmy=_CGMY_COMMON,
    sigma_const=None,
    beta_s_theory=-0.4,
    beta_c_theory=float("nan"),  # not used in Exp 1 & 2
)

SCENARIO_S2 = ScenarioConfig(
    name="S2",
    label="Jump-dominated skew",
    description="H=0.35, Y=1.5 => beta_s=min(-0.15,-0.25)=-0.25 (jumps dominate)",
    rh=RoughHestonParams(V0=0.04, vol_of_vol=0.2, kappa=1.5, theta=0.04, rho=-0.1, H=0.35),
    cgmy=_CGMY_COMMON,
    sigma_const=None,
    beta_s_theory=-0.25,
    beta_c_theory=float("nan"),
)

SCENARIO_S3 = ScenarioConfig(
    name="S3",
    label="Critical skew line",
    description="H=0.25=1-Y/2 with Y=1.5 => beta_s=-0.25 (both contribute)",
    rh=RoughHestonParams(V0=0.04, vol_of_vol=0.45, kappa=1.5, theta=0.04, rho=-0.7, H=0.25),
    cgmy=_CGMY_COMMON,
    sigma_const=None,
    beta_s_theory=-0.25,
    beta_c_theory=float("nan"),
)

SKEW_SCENARIOS = [SCENARIO_S1, SCENARIO_S2, SCENARIO_S3]


# ---------- Exp 3 scenarios ----------
_CGMY_EXP3 = CGMYParams(C_plus=0.015, C_minus=0.025, G=5.0, M=8.0, Y=1.5)

SCENARIO_C1 = ScenarioConfig(
    name="C1",
    label="Rough-only (no jumps)",
    description="H=0.1, L=0 => beta_s=-0.4, beta_c=-0.8 => beta_c=2*beta_s",
    rh=RoughHestonParams(V0=0.04, vol_of_vol=0.45, kappa=1.5, theta=0.04, rho=-0.5, H=0.1),
    cgmy=None,
    sigma_const=None,
    beta_s_theory=-0.4,
    beta_c_theory=-0.8,
)

SCENARIO_C2 = ScenarioConfig(
    name="C2",
    label="Pure jump + constant vol",
    description="sigma=0.2 + CGMY, no rough Heston => beta_s=-0.25, beta_c=-0.75",
    rh=None,
    cgmy=_CGMY_EXP3,
    sigma_const=0.2,
    beta_s_theory=-0.25,
    beta_c_theory=-0.75,
)

SCENARIO_C3 = ScenarioConfig(
    name="C3",
    label="Mixed rough Heston + CGMY",
    description=(
        "H=0.2, Y=1.5 => beta_s=min(-0.3,-0.25)=-0.3, "
        "beta_c=min(-0.6,-0.75)=-0.75 (breaks both pure-model relations)"
    ),
    rh=RoughHestonParams(V0=0.04, vol_of_vol=0.45, kappa=1.5, theta=0.04, rho=-0.7, H=0.2),
    cgmy=_CGMY_EXP3,
    sigma_const=None,
    beta_s_theory=-0.3,
    beta_c_theory=-0.75,
)

CURVATURE_SCENARIOS = [SCENARIO_C1, SCENARIO_C2, SCENARIO_C3]


def make_maturity_grid(
    T_min: float = 1e-11,
    T_max: float = 1e-3,
    N: int = 80,
) -> np.ndarray:
    """
    Build a log-spaced maturity grid over [T_min, T_max].

    Parameters
    ----------
    T_min : float  smallest maturity
    T_max : float  largest maturity
    N : int        number of grid points

    Returns
    -------
    ndarray, shape (N,)
    """
    return np.logspace(np.log10(T_min), np.log10(T_max), N)
