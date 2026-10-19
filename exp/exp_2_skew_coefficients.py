"""
Experiment 2: ATM implied-volatility skew leading coefficients.

Replicates the paper's asymptotic prefactor A_s for:
  S1 rough-dominated: A_s ≈ -0.551
  S2 jump-dominated:  A_s ≈ -0.154
  S3 critical line:   A_s ≈ -0.643

Computes hat_A_s(T) = skew(T) / T^{beta_s_theory} and checks convergence.

Results saved to results/exp2_*.csv and results/exp2_skew_coefficients.png
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

from src.engine import run_scenario, compute_skew_coefficients, print_summary
from src.scenarios import SKEW_SCENARIOS, make_maturity_grid
from src.atm_quantities import (
    rough_dominated_skew_coefficient,
    jump_dominated_skew_coefficient,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Paper's approximate target values for the three scenarios
PAPER_TARGETS = {"S1": -0.551, "S2": -0.154, "S3": -0.643}


def _compute_theoretical_A_s(scenario) -> float:
    """Compute A_s from formula based on regime."""
    rh = scenario.rh
    cgmy = scenario.cgmy

    beta_s_rough = rh.H - 0.5 if rh is not None else -np.inf
    beta_s_jump = (1.0 - cgmy.Y) / 2.0 if cgmy is not None else -np.inf

    if rh is None:
        A_rough = 0.0
    else:
        A_rough = rough_dominated_skew_coefficient(
            rh.V0, rh.vol_of_vol, rh.rho, rh.H
        )

    if cgmy is None:
        A_jump = 0.0
    else:
        A_jump = jump_dominated_skew_coefficient(
            rh.V0 if rh else 0.04,
            cgmy.C_plus, cgmy.C_minus, cgmy.Y,
        )

    if abs(beta_s_rough - beta_s_jump) < 1e-6:
        # Critical: both contribute
        return A_rough + A_jump
    elif beta_s_rough < beta_s_jump:
        return A_rough
    else:
        return A_jump


def run_exp2(
    N_T: int = 80,
    N_steps: int = 100,
    N_quad: int = 300,
    T_min: float = 1e-8,
    T_max: float = 1e-3,
) -> dict[str, pd.DataFrame]:
    """
    Run Experiment 2 for all three skew scenarios.

    Parameters
    ----------
    N_T, N_steps, N_quad, T_min, T_max : int/float  grid parameters

    Returns
    -------
    dict mapping scenario name -> DataFrame
    """
    T_grid = make_maturity_grid(T_min, T_max, N_T)
    results = {}

    for scenario in SKEW_SCENARIOS:
        df = run_scenario(scenario, T_grid, N_steps=N_steps, N_quad=N_quad, verbose=True)
        df_coeff = compute_skew_coefficients(df, scenario)

        # Override A_s_theory with the direct formula
        A_s = _compute_theoretical_A_s(scenario)
        df_coeff["A_s_theory"] = A_s
        df_coeff["abs_dev"] = np.abs(df_coeff["hat_A_s"] - A_s)

        df_coeff.to_csv(
            os.path.join(RESULTS_DIR, f"exp2_{scenario.name}.csv"), index=False
        )
        results[scenario.name] = df_coeff

        print(f"\n  Scenario {scenario.name}:")
        print(f"    A_s_theory (formula) = {A_s:.4f}")
        print(f"    Paper target         ≈ {PAPER_TARGETS[scenario.name]:.4f}")
        valid = df_coeff.dropna(subset=["hat_A_s"])
        if len(valid) > 0:
            n_avg = min(5, len(valid))
            hat_mean = valid["hat_A_s"].iloc[:n_avg].mean()
            print(f"    hat_A_s (avg 5 smallest T) = {hat_mean:.4f}")

    return results


def plot_exp2(results: dict[str, pd.DataFrame]) -> None:
    """Plot hat_A_s(T) vs T for all scenarios."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        r"Exp 2: Normalised Skew Coefficient  $\hat{A}_s(T)$ vs $T$",
        fontsize=13,
    )

    for ax, scenario in zip(axes, SKEW_SCENARIOS):
        df = results[scenario.name]
        valid = df.dropna(subset=["hat_A_s"])
        A_theory = _compute_theoretical_A_s(scenario)
        paper_target = PAPER_TARGETS[scenario.name]

        ax.semilogx(
            valid["T"], valid["hat_A_s"],
            "o-", markersize=3, linewidth=1.2, label=r"$\hat{A}_s(T)$",
        )
        ax.axhline(
            A_theory, color="blue", linestyle="--",
            linewidth=1.5, label=f"formula A_s={A_theory:.3f}",
        )
        ax.axhline(
            paper_target, color="red", linestyle=":",
            linewidth=1.5, label=f"paper ≈{paper_target:.3f}",
        )
        ax.set_xlabel("Maturity T")
        ax.set_ylabel(r"$\hat{A}_s = \partial_k I / T^{\beta_s}$")
        ax.set_title(f"{scenario.name}: {scenario.label}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "exp2_skew_coefficients.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {out_path}")


def print_exp2_table(results: dict[str, pd.DataFrame]) -> None:
    """Print coefficient convergence table."""
    print("\n" + "=" * 75)
    print("Exp 2 Summary — Skew Leading Coefficient Convergence")
    print("=" * 75)
    header = (
        f"{'Scenario':<10} {'A_s_formula':>14} {'A_s_paper':>12}"
        f" {'hat_A_s_min5T':>15} {'diff_formula':>14}"
    )
    print(header)
    print("-" * 75)
    for sc in SKEW_SCENARIOS:
        df = results[sc.name]
        A_formula = _compute_theoretical_A_s(sc)
        A_paper = PAPER_TARGETS[sc.name]
        valid = df.dropna(subset=["hat_A_s"])
        if len(valid) == 0:
            continue
        n_avg = min(5, len(valid))
        hat_mean = valid["hat_A_s"].iloc[:n_avg].mean()
        diff = hat_mean - A_formula
        print(
            f"{sc.name:<10} {A_formula:>14.4f} {A_paper:>12.4f}"
            f" {hat_mean:>15.4f} {diff:>14.4f}"
        )
    print("=" * 75)


if __name__ == "__main__":
    print("Running Experiment 2: ATM Skew Leading Coefficients")
    results = run_exp2(N_T=80, N_steps=100, N_quad=300, T_min=1e-8, T_max=1e-3)
    plot_exp2(results)
    print_exp2_table(results)
    print("\nExperiment 2 complete.")
