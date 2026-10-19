"""
Experiment 1: ATM implied-volatility skew power-law exponents.

Replicates the paper's short-maturity asymptotic results for the ATM skew
in mixed rough Heston + CGMY models, confirming:
  - Power-law form: dI/dk ~ A_s * T^{beta_s}
  - Exponent: beta_s = min(H - 1/2, (1 - Y)/2)
  - Three regimes: rough-dominated (S1), jump-dominated (S2), critical (S3)

Results saved to results/exp1_*.csv and results/exp1_skew_power.png
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.engine import run_scenario, print_summary
from src.scenarios import SKEW_SCENARIOS, make_maturity_grid

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_exp1(
    N_T: int = 80,
    N_steps: int = 100,
    N_quad: int = 300,
    T_min: float = 1e-8,
    T_max: float = 1e-3,
) -> dict[str, pd.DataFrame]:
    """
    Run Experiment 1 for all three skew scenarios.

    Parameters
    ----------
    N_T : int       number of maturity grid points
    N_steps : int   fractional Riccati time steps
    N_quad : int    quadrature nodes for vectorised Gil-Pelaez inversion
    T_min : float   smallest maturity
    T_max : float   largest maturity

    Returns
    -------
    dict mapping scenario name -> DataFrame of results
    """
    T_grid = make_maturity_grid(T_min, T_max, N_T)
    results = {}

    for scenario in SKEW_SCENARIOS:
        df = run_scenario(scenario, T_grid, N_steps=N_steps, N_quad=N_quad, verbose=True)
        df.to_csv(
            os.path.join(RESULTS_DIR, f"exp1_{scenario.name}.csv"), index=False
        )
        print_summary(df, scenario)
        results[scenario.name] = df

    return results


def plot_exp1(results: dict[str, pd.DataFrame]) -> None:
    """Plot local skew exponent estimates vs log(T) for all scenarios."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Exp 1: ATM Skew Power-Law Exponent  —  hat_beta_s(T) vs log(T)",
        fontsize=13,
    )

    for ax, scenario in zip(axes, SKEW_SCENARIOS):
        df = results[scenario.name]
        valid = df.dropna(subset=["hat_beta_s"])

        ax.semilogx(
            valid["T"], valid["hat_beta_s"],
            "o-", markersize=3, linewidth=1.2, label=r"$\hat{\beta}_s(T)$",
        )
        ax.axhline(
            scenario.beta_s_theory, color="red", linestyle="--",
            linewidth=1.5, label=f"theory={scenario.beta_s_theory:.2f}",
        )
        ax.set_xlabel("Maturity T")
        ax.set_ylabel(r"Local slope $\hat{\beta}_s$")
        ax.set_title(f"{scenario.name}: {scenario.label}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-1.5, 0.5])

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "exp1_skew_power.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {out_path}")


def print_exp1_table(results: dict[str, pd.DataFrame]) -> None:
    """Print convergence table with smallest-maturity slope estimates."""
    print("\n" + "=" * 65)
    print("Exp 1 Summary — smallest-maturity local slope estimates")
    print("=" * 65)
    header = f"{'Scenario':<10} {'beta_s_theory':>14} {'hat_beta_s (min T)':>20} {'diff':>10}"
    print(header)
    print("-" * 65)
    for sc in SKEW_SCENARIOS:
        df = results[sc.name]
        valid = df.dropna(subset=["hat_beta_s"])
        if len(valid) == 0:
            print(f"  {sc.name}: no valid slopes.")
            continue
        # Use 5 smallest-T slopes and average
        n_avg = min(5, len(valid))
        hat_mean = valid["hat_beta_s"].iloc[:n_avg].mean()
        diff = hat_mean - sc.beta_s_theory
        print(
            f"{sc.name:<10} {sc.beta_s_theory:>14.4f} {hat_mean:>20.4f} {diff:>10.4f}"
        )
    print("=" * 65)


if __name__ == "__main__":
    print("Running Experiment 1: ATM Skew Power-Law Exponents")
    results = run_exp1(N_T=80, N_steps=100, N_quad=300, T_min=1e-8, T_max=1e-3)
    plot_exp1(results)
    print_exp1_table(results)
    print("\nExperiment 1 complete.")
