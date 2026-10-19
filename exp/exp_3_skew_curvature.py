"""
Experiment 3: ATM IV skew-curvature power-law relations.

Tests the joint exponent relations across three model classes:
  C1 rough-only (H=0.1, L=0):          beta_c = 2*beta_s  (-0.8 = 2*(-0.4))
  C2 pure jump + const vol (Y=1.5):    beta_c = beta_s - 1/2 (-0.75 = -0.25 - 0.5)
  C3 mixed rough Heston + CGMY:        neither relation holds (beta_s=-0.3, beta_c=-0.75)

Paper Table 1 smallest-maturity targets:
  C1: (hat_beta_s, hat_beta_c) ≈ (-0.4001, -0.7978)
  C2: (hat_beta_s, hat_beta_c) ≈ (-0.2514, -0.7520)
  C3: (hat_beta_s, hat_beta_c) ≈ (-0.2969, -0.7531)

Results saved to results/exp3_*.csv and results/exp3_skew_curvature.png
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
from src.scenarios import CURVATURE_SCENARIOS, make_maturity_grid

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Paper Table 1 targets
PAPER_TABLE1 = {
    "C1": (-0.4001, -0.7978),
    "C2": (-0.2514, -0.7520),
    "C3": (-0.2969, -0.7531),
}


def run_exp3(
    N_T: int = 80,
    N_steps: int = 100,
    N_quad: int = 300,
    T_min: float = 1e-8,
    T_max: float = 1e-3,
) -> dict[str, pd.DataFrame]:
    """
    Run Experiment 3 for all three skew-curvature scenarios.

    Parameters
    ----------
    N_T, N_steps, N_quad, T_min, T_max : grid parameters

    Returns
    -------
    dict mapping scenario name -> DataFrame
    """
    T_grid = make_maturity_grid(T_min, T_max, N_T)
    results = {}

    for scenario in CURVATURE_SCENARIOS:
        df = run_scenario(scenario, T_grid, N_steps=N_steps, N_quad=N_quad, verbose=True)
        df.to_csv(
            os.path.join(RESULTS_DIR, f"exp3_{scenario.name}.csv"), index=False
        )
        print_summary(df, scenario)
        results[scenario.name] = df

    return results


def plot_exp3(results: dict[str, pd.DataFrame]) -> None:
    """Plot hat_beta_s and hat_beta_c vs log(T) for all scenarios."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        "Exp 3: ATM Skew and Curvature Power-Law Exponents vs Maturity",
        fontsize=13,
    )

    for col, scenario in enumerate(CURVATURE_SCENARIOS):
        df = results[scenario.name]
        paper_bs, paper_bc = PAPER_TABLE1[scenario.name]

        # Top row: skew exponent
        ax_s = axes[0, col]
        valid_s = df.dropna(subset=["hat_beta_s"])
        ax_s.semilogx(
            valid_s["T"], valid_s["hat_beta_s"],
            "o-", markersize=3, linewidth=1.2, label=r"$\hat{\beta}_s$",
        )
        ax_s.axhline(
            scenario.beta_s_theory, color="red", linestyle="--",
            label=f"theory={scenario.beta_s_theory:.2f}",
        )
        ax_s.axhline(
            paper_bs, color="purple", linestyle=":",
            label=f"paper≈{paper_bs:.4f}",
        )
        ax_s.set_title(f"{scenario.name}: {scenario.label}")
        ax_s.set_ylabel(r"$\hat{\beta}_s$")
        ax_s.legend(fontsize=8)
        ax_s.grid(True, alpha=0.3)
        ax_s.set_ylim([-1.5, 0.5])

        # Bottom row: curvature exponent
        ax_c = axes[1, col]
        valid_c = df.dropna(subset=["hat_beta_c"])
        ax_c.semilogx(
            valid_c["T"], valid_c["hat_beta_c"],
            "s-", markersize=3, linewidth=1.2, color="orange",
            label=r"$\hat{\beta}_c$",
        )
        ax_c.axhline(
            scenario.beta_c_theory, color="red", linestyle="--",
            label=f"theory={scenario.beta_c_theory:.2f}",
        )
        ax_c.axhline(
            paper_bc, color="purple", linestyle=":",
            label=f"paper≈{paper_bc:.4f}",
        )
        ax_c.set_xlabel("Maturity T")
        ax_c.set_ylabel(r"$\hat{\beta}_c$")
        ax_c.legend(fontsize=8)
        ax_c.grid(True, alpha=0.3)
        ax_c.set_ylim([-1.5, 0.5])

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "exp3_skew_curvature.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {out_path}")


def print_exp3_table(results: dict[str, pd.DataFrame]) -> None:
    """Print convergence table for both skew and curvature exponents."""
    print("\n" + "=" * 85)
    print("Exp 3 Summary — Skew & Curvature Exponent Convergence (smallest-T averages)")
    print("=" * 85)
    header = (
        f"{'Scen':<6} "
        f"{'bs_theory':>10} {'hat_bs':>10} {'bc_theory':>10} {'hat_bc':>10} "
        f"{'paper_bs':>10} {'paper_bc':>10} "
        f"{'rel_C1':>12}"
    )
    print(header)
    print("-" * 85)

    for sc in CURVATURE_SCENARIOS:
        df = results[sc.name]
        paper_bs, paper_bc = PAPER_TABLE1[sc.name]

        vs = df.dropna(subset=["hat_beta_s"])
        vc = df.dropna(subset=["hat_beta_c"])

        n_avg = 5
        hat_bs = vs["hat_beta_s"].iloc[:min(n_avg, len(vs))].mean() if len(vs) > 0 else np.nan
        hat_bc = vc["hat_beta_c"].iloc[:min(n_avg, len(vc))].mean() if len(vc) > 0 else np.nan

        # Check which pure-model relation holds
        if not np.isnan(hat_bs) and not np.isnan(hat_bc):
            rel_c1 = f"bc/bs={hat_bc/hat_bs:.3f}" if abs(hat_bs) > 1e-6 else "—"
        else:
            rel_c1 = "—"

        print(
            f"{sc.name:<6} "
            f"{sc.beta_s_theory:>10.4f} {hat_bs:>10.4f} "
            f"{sc.beta_c_theory:>10.4f} {hat_bc:>10.4f} "
            f"{paper_bs:>10.4f} {paper_bc:>10.4f} "
            f"{rel_c1:>12}"
        )
    print("=" * 85)

    print("\nRelations:")
    print("  C1 (rough-only):  beta_c = 2*beta_s => bc/bs should be 2.0")
    print("  C2 (pure jump):   beta_c = beta_s - 0.5 => bc - bs should be -0.5")
    print("  C3 (mixed):       neither relation satisfied")


if __name__ == "__main__":
    print("Running Experiment 3: ATM Skew-Curvature Power-Law Relations")
    results = run_exp3(N_T=80, N_steps=100, N_quad=300, T_min=1e-8, T_max=1e-3)
    plot_exp3(results)
    print_exp3_table(results)
    print("\nExperiment 3 complete.")
