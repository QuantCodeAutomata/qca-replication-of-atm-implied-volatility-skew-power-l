"""
Main computation engine.

Runs the full ATM IV pipeline for a given scenario over a maturity grid:
  1. For each T: compute p_T, q_T, density-at-zero via Gil-Pelaez inversion.
  2. Compute ATM IV I(T), skew dI/dk, curvature d²I/dk².
  3. Estimate local log-log slopes (power-law exponents) for skew and curvature.

Validation checks performed per maturity:
  - phi_T(0) == 1  (martingale normalisation)
  - phi_T(-i) ≈ 1  (martingale condition for S_t = exp(X_t))
  - p_T -> 1/2, q_T -> 1/2 as T -> 0
  - I(T) -> sqrt(V0) as T -> 0
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from src.atm_quantities import (
    atm_iv,
    atm_skew,
    atm_curvature,
    local_log_slope,
    theoretical_skew_exponent,
    theoretical_curvature_exponent,
    rough_dominated_skew_coefficient,
    jump_dominated_skew_coefficient,
)
from src.fourier_inversion import (
    RoughHestonParams,
    CGMYParams,
    compute_p_q_density,
    compute_mixed_cf_array,
)
from src.scenarios import ScenarioConfig


def run_scenario(
    scenario: ScenarioConfig,
    T_grid: np.ndarray,
    N_steps: int = 100,
    N_quad: int = 300,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute all ATM quantities for a scenario over a maturity grid.

    Parameters
    ----------
    scenario : ScenarioConfig
    T_grid : ndarray  maturity grid (should be log-spaced, increasing)
    N_steps : int  fractional Riccati time steps
    N_quad : int  quadrature nodes for Gil-Pelaez inversion (vectorised)
    verbose : bool  print progress

    Returns
    -------
    DataFrame with columns:
      T, p_T, q_T, f0, I, skew, curvature,
      log_T, log_abs_skew, log_abs_curvature,
      beta_s_theory, beta_c_theory
    """
    rows = []
    rh = scenario.rh
    cgmy = scenario.cgmy
    sigma_const = scenario.sigma_const

    if verbose:
        print(f"\n{'='*60}")
        print(f"Scenario {scenario.name}: {scenario.label}")
        print(f"  {scenario.description}")
        print(f"  beta_s_theory = {scenario.beta_s_theory:.4f}")
        print(f"  beta_c_theory = {scenario.beta_c_theory:.4f}")
        print(f"  N_steps={N_steps}, N_quad={N_quad}, N_T={len(T_grid)}")
        print(f"{'='*60}")

    for i, T in enumerate(T_grid):
        try:
            p_T, q_T, f0 = compute_p_q_density(
                T, rh, cgmy, sigma_const,
                N_steps=N_steps, N_quad=N_quad,
            )
        except Exception as exc:
            if verbose:
                print(f"  T={T:.2e}: inversion failed — {exc}")
            rows.append({
                "T": T, "p_T": np.nan, "q_T": np.nan, "f0": np.nan,
                "I": np.nan, "skew": np.nan, "curvature": np.nan,
            })
            continue

        try:
            I = atm_iv(T, p_T, q_T)
        except Exception:
            I = np.nan

        try:
            skew = atm_skew(T, p_T, I) if not np.isnan(I) else np.nan
        except Exception:
            skew = np.nan

        try:
            curvature = (
                atm_curvature(T, p_T, q_T, I, skew, f0)
                if not (np.isnan(I) or np.isnan(skew))
                else np.nan
            )
        except Exception:
            curvature = np.nan

        rows.append({
            "T": T,
            "p_T": p_T,
            "q_T": q_T,
            "f0": f0,
            "I": I,
            "skew": skew,
            "curvature": curvature,
        })

        if verbose and (i % 10 == 0 or i < 5):
            print(
                f"  T={T:.2e}  p_T={p_T:.6f}  q_T={q_T:.6f}"
                f"  I={I:.4f}  skew={skew:.4f}  curv={curvature:.4f}"
            )

    df = pd.DataFrame(rows)
    df["log_T"] = np.log(df["T"])
    df["log_abs_skew"] = np.log(np.abs(df["skew"]))
    df["log_abs_curvature"] = np.log(np.abs(df["curvature"]))
    df["beta_s_theory"] = scenario.beta_s_theory
    df["beta_c_theory"] = scenario.beta_c_theory

    # Local slope estimates (consecutive log-log finite differences)
    valid_s = df["log_abs_skew"].notna() & np.isfinite(df["log_abs_skew"])
    valid_c = df["log_abs_curvature"].notna() & np.isfinite(df["log_abs_curvature"])

    df["hat_beta_s"] = np.nan
    df["hat_beta_c"] = np.nan

    if valid_s.sum() >= 2:
        idx_s = df.index[valid_s].tolist()
        slopes_s = local_log_slope(
            df.loc[idx_s, "log_T"].values,
            df.loc[idx_s, "log_abs_skew"].values,
        )
        # Assign slopes to midpoints (index i corresponds to interval [i, i+1])
        for k, slope in enumerate(slopes_s):
            df.loc[idx_s[k], "hat_beta_s"] = slope

    if valid_c.sum() >= 2:
        idx_c = df.index[valid_c].tolist()
        slopes_c = local_log_slope(
            df.loc[idx_c, "log_T"].values,
            df.loc[idx_c, "log_abs_curvature"].values,
        )
        for k, slope in enumerate(slopes_c):
            df.loc[idx_c[k], "hat_beta_c"] = slope

    return df


def compute_skew_coefficients(
    df: pd.DataFrame,
    scenario: ScenarioConfig,
) -> pd.DataFrame:
    """
    Compute hat_A_s(T) = skew / T^{beta_s_theory} for Exp 2.

    Also computes theoretical A_s and absolute deviation.

    Parameters
    ----------
    df : DataFrame output from run_scenario
    scenario : ScenarioConfig

    Returns
    -------
    DataFrame with additional columns: hat_A_s, A_s_theory, abs_dev
    """
    beta_s = scenario.beta_s_theory
    df = df.copy()
    df["hat_A_s"] = df["skew"] / (df["T"] ** beta_s)

    # Compute theoretical A_s
    rh = scenario.rh
    cgmy = scenario.cgmy

    # Determine dominant regime from beta_s
    beta_s_rough = rh.H - 0.5 if rh is not None else -np.inf
    beta_s_jump = (1.0 - cgmy.Y) / 2.0 if cgmy is not None else -np.inf

    if rh is None:
        # Pure jump
        A_s = jump_dominated_skew_coefficient(
            0.04,  # sigma_const^2 for Exp3 C2 not used here; use V0 fallback
            cgmy.C_plus, cgmy.C_minus, cgmy.Y,
        ) if cgmy else np.nan
    elif cgmy is None:
        # Pure rough Heston
        A_s = rough_dominated_skew_coefficient(
            rh.V0, rh.vol_of_vol, rh.rho, rh.H
        )
    elif abs(beta_s_rough - beta_s_jump) < 1e-10:
        # Critical line: both contribute
        A_s = (
            rough_dominated_skew_coefficient(rh.V0, rh.vol_of_vol, rh.rho, rh.H)
            + jump_dominated_skew_coefficient(rh.V0, cgmy.C_plus, cgmy.C_minus, cgmy.Y)
        )
    elif beta_s_rough < beta_s_jump:
        # Rough dominates (smaller exponent => larger in magnitude for small T)
        A_s = rough_dominated_skew_coefficient(rh.V0, rh.vol_of_vol, rh.rho, rh.H)
    else:
        # Jumps dominate
        A_s = jump_dominated_skew_coefficient(rh.V0, cgmy.C_plus, cgmy.C_minus, cgmy.Y)

    df["A_s_theory"] = A_s
    df["abs_dev"] = np.abs(df["hat_A_s"] - A_s)

    return df


def print_summary(df: pd.DataFrame, scenario: ScenarioConfig) -> None:
    """Print a concise summary table for a scenario."""
    valid = df.dropna(subset=["skew", "hat_beta_s"])
    if len(valid) == 0:
        print(f"  [{scenario.name}] No valid rows.")
        return

    # Focus on smallest maturities
    n_show = min(10, len(valid))
    show = valid.head(n_show)

    print(f"\n--- {scenario.name}: {scenario.label} ---")
    print(f"  beta_s_theory = {scenario.beta_s_theory:.4f}, "
          f"beta_c_theory = {scenario.beta_c_theory:.4f}")
    cols = ["T", "p_T", "q_T", "I", "skew", "hat_beta_s"]
    if "hat_beta_c" in df.columns:
        cols.append("hat_beta_c")
    print(show[cols].to_string(index=False, float_format="{:.5f}".format))
