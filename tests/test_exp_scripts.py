"""
Tests that verify the experiment scripts can be imported and executed
with small grids without errors.

Uses minimal parameters for speed:
  N_T=5, N_steps=20, N_quad=50, T_min=1e-5, T_max=1e-3
"""

import warnings

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Small-grid parameters shared across all experiments
# ---------------------------------------------------------------------------
SMALL_GRID = dict(N_T=5, N_steps=20, N_quad=50, T_min=1e-5, T_max=1e-3)


# ===========================================================================
# Experiment 1: ATM skew power-law exponents
# ===========================================================================


def test_exp1_imports():
    """Experiment 1 module can be imported without errors."""
    import exp.exp_1_skew_power  # noqa: F401


def test_exp1_run_returns_dict():
    """run_exp1 with a small grid returns a dict of DataFrames."""
    from exp.exp_1_skew_power import run_exp1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_exp1(**SMALL_GRID)

    assert isinstance(results, dict), f"Expected dict, got {type(results)}"


def test_exp1_run_has_all_scenarios():
    """run_exp1 returns results for all three skew scenarios: S1, S2, S3."""
    from exp.exp_1_skew_power import run_exp1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_exp1(**SMALL_GRID)

    assert set(results.keys()) == {"S1", "S2", "S3"}, (
        f"Expected keys S1/S2/S3, got {set(results.keys())}"
    )


def test_exp1_each_result_is_dataframe():
    """Each value in run_exp1 results is a DataFrame with expected columns."""
    import pandas as pd
    from exp.exp_1_skew_power import run_exp1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_exp1(**SMALL_GRID)

    required_cols = {"T", "p_T", "q_T", "I", "skew"}
    for name, df in results.items():
        assert isinstance(df, pd.DataFrame), (
            f"Scenario {name}: expected DataFrame, got {type(df)}"
        )
        missing = required_cols - set(df.columns)
        assert not missing, f"Scenario {name}: missing columns {missing}"


def test_exp1_result_row_count():
    """run_exp1 produces exactly N_T rows per scenario."""
    from exp.exp_1_skew_power import run_exp1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_exp1(**SMALL_GRID)

    for name, df in results.items():
        assert len(df) == SMALL_GRID["N_T"], (
            f"Scenario {name}: expected {SMALL_GRID['N_T']} rows, got {len(df)}"
        )


def test_exp1_plot_does_not_raise():
    """plot_exp1 runs without raising an exception."""
    from exp.exp_1_skew_power import plot_exp1, run_exp1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_exp1(**SMALL_GRID)
        plot_exp1(results)


def test_exp1_print_table_does_not_raise(capsys):
    """print_exp1_table runs and produces output containing scenario names."""
    from exp.exp_1_skew_power import print_exp1_table, run_exp1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_exp1(**SMALL_GRID)
    print_exp1_table(results)
    captured = capsys.readouterr()
    for name in ("S1", "S2", "S3"):
        assert name in captured.out, f"Scenario {name} not found in print_exp1_table output"


# ===========================================================================
# Experiment 2: ATM skew leading coefficients
# ===========================================================================


def test_exp2_imports():
    """Experiment 2 module can be imported without errors."""
    import exp.exp_2_skew_coefficients  # noqa: F401


def test_exp2_run_returns_dict():
    """run_exp2 with a small grid returns a dict of DataFrames."""
    from exp.exp_2_skew_coefficients import run_exp2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_exp2(**SMALL_GRID)

    assert isinstance(results, dict), f"Expected dict, got {type(results)}"


def test_exp2_run_has_all_scenarios():
    """run_exp2 returns results for all three scenarios: S1, S2, S3."""
    from exp.exp_2_skew_coefficients import run_exp2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_exp2(**SMALL_GRID)

    assert set(results.keys()) == {"S1", "S2", "S3"}, (
        f"Expected keys S1/S2/S3, got {set(results.keys())}"
    )


def test_exp2_result_has_hat_As_column():
    """Each DataFrame in run_exp2 results contains the hat_A_s coefficient column."""
    from exp.exp_2_skew_coefficients import run_exp2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_exp2(**SMALL_GRID)

    for name, df in results.items():
        assert "hat_A_s" in df.columns, (
            f"Scenario {name}: 'hat_A_s' column missing"
        )


def test_exp2_plot_does_not_raise():
    """plot_exp2 runs without raising an exception."""
    from exp.exp_2_skew_coefficients import plot_exp2, run_exp2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_exp2(**SMALL_GRID)
        plot_exp2(results)


def test_exp2_print_table_does_not_raise(capsys):
    """print_exp2_table runs and produces output."""
    from exp.exp_2_skew_coefficients import print_exp2_table, run_exp2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_exp2(**SMALL_GRID)
    print_exp2_table(results)
    captured = capsys.readouterr()
    assert len(captured.out) > 0, "print_exp2_table produced no output"


def test_exp2_hat_As_sign():
    """hat_A_s values should be negative for all negatively-skewed scenarios."""
    from exp.exp_2_skew_coefficients import run_exp2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_exp2(**SMALL_GRID)

    for name, df in results.items():
        valid = df["hat_A_s"].dropna()
        if len(valid) == 0:
            continue
        assert (valid < 0).all(), (
            f"Scenario {name}: hat_A_s values should be negative, got {valid.values}"
        )


# ===========================================================================
# Experiment 3: ATM skew-curvature power-law relations
# ===========================================================================


def test_exp3_imports():
    """Experiment 3 module can be imported without errors."""
    import exp.exp_3_skew_curvature  # noqa: F401


def test_exp3_run_returns_dict():
    """run_exp3 with a small grid returns a dict of DataFrames."""
    from exp.exp_3_skew_curvature import run_exp3

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_exp3(**SMALL_GRID)

    assert isinstance(results, dict), f"Expected dict, got {type(results)}"


def test_exp3_run_has_all_scenarios():
    """run_exp3 returns results for all three curvature scenarios: C1, C2, C3."""
    from exp.exp_3_skew_curvature import run_exp3

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_exp3(**SMALL_GRID)

    assert set(results.keys()) == {"C1", "C2", "C3"}, (
        f"Expected keys C1/C2/C3, got {set(results.keys())}"
    )


def test_exp3_each_result_has_curvature_column():
    """Each DataFrame in run_exp3 results contains the curvature column."""
    from exp.exp_3_skew_curvature import run_exp3

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_exp3(**SMALL_GRID)

    for name, df in results.items():
        assert "curvature" in df.columns, (
            f"Scenario {name}: 'curvature' column missing"
        )


def test_exp3_result_row_count():
    """run_exp3 produces exactly N_T rows per scenario."""
    from exp.exp_3_skew_curvature import run_exp3

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_exp3(**SMALL_GRID)

    for name, df in results.items():
        assert len(df) == SMALL_GRID["N_T"], (
            f"Scenario {name}: expected {SMALL_GRID['N_T']} rows, got {len(df)}"
        )


def test_exp3_plot_does_not_raise():
    """plot_exp3 runs without raising an exception."""
    from exp.exp_3_skew_curvature import plot_exp3, run_exp3

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_exp3(**SMALL_GRID)
        plot_exp3(results)


def test_exp3_print_table_does_not_raise(capsys):
    """print_exp3_table runs and produces output."""
    from exp.exp_3_skew_curvature import print_exp3_table, run_exp3

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_exp3(**SMALL_GRID)
    print_exp3_table(results)
    captured = capsys.readouterr()
    for name in ("C1", "C2", "C3"):
        assert name in captured.out, f"Scenario {name} not found in print_exp3_table output"


def test_exp3_c1_curvature_exponent_relation():
    """For C1 (rough-only), hat_beta_c ≈ 2*hat_beta_s (rough-only power-law identity)."""
    from exp.exp_3_skew_curvature import run_exp3

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_exp3(**SMALL_GRID)

    df_c1 = results["C1"]
    valid = df_c1.dropna(subset=["hat_beta_s", "hat_beta_c"])
    if len(valid) == 0:
        pytest.skip("No valid slope estimates for C1 with this small grid")

    # For rough-only model: beta_c = 2*beta_s, so hat_beta_c / hat_beta_s ≈ 2
    # Use the first (smallest-T) valid row
    row = valid.iloc[0]
    ratio = row["hat_beta_c"] / row["hat_beta_s"]
    assert abs(ratio - 2.0) < 0.5, (
        f"C1: hat_beta_c/hat_beta_s = {ratio:.3f}, expected close to 2.0"
    )
