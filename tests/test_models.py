"""
Comprehensive pytest tests for the rough Heston + CGMY model.

Covers mathematical properties of all modules:
  - src/cgmy.py          : CGMY exponent, drift compensation, complex power
  - src/rough_heston.py  : fractional Riccati solver, rough Heston CF
  - src/fourier_inversion.py : Gil-Pelaez inversion, mixed model CF
  - src/atm_quantities.py : ATM IV/skew/curvature, theoretical exponents
  - src/scenarios.py     : maturity grid, scenario definitions
  - src/engine.py        : run_scenario, compute_skew_coefficients, print_summary
"""

import warnings

import numpy as np
import pytest

from src.atm_quantities import (
    atm_curvature,
    atm_iv,
    atm_skew,
    jump_dominated_skew_coefficient,
    local_log_slope,
    rough_dominated_skew_coefficient,
    theoretical_curvature_exponent,
    theoretical_skew_exponent,
)
from src.cgmy import _cpow, cgmy_drift_compensation, cgmy_exponent
from src.engine import compute_skew_coefficients, print_summary, run_scenario
from src.fourier_inversion import (
    CGMYParams,
    RoughHestonParams,
    compute_mixed_cf_array,
    compute_p_q_density,
)
from src.rough_heston import rough_heston_cf, solve_fractional_riccati
from src.scenarios import (
    CURVATURE_SCENARIOS,
    SKEW_SCENARIOS,
    make_maturity_grid,
)

# ---------------------------------------------------------------------------
# Shared parameter sets (small N_steps/N_quad for fast execution)
# ---------------------------------------------------------------------------

# Standard CGMY parameters (from paper)
C_PLUS = 0.015
C_MINUS = 0.025
G = 5.0
M = 8.0
Y = 1.5

# Standard rough Heston parameters (Scenario S1)
V0 = 0.04
VOL_OF_VOL = 0.45
KAPPA = 1.5
THETA = 0.04
RHO = -0.7
H = 0.1

# Fast computation settings
FAST = dict(N_steps=20, N_quad=50)


def _make_rh() -> RoughHestonParams:
    return RoughHestonParams(
        V0=V0, vol_of_vol=VOL_OF_VOL, kappa=KAPPA, theta=THETA, rho=RHO, H=H
    )


def _make_cgmy() -> CGMYParams:
    return CGMYParams(C_plus=C_PLUS, C_minus=C_MINUS, G=G, M=M, Y=Y)


def _heston_cf_analytical(
    z_array: np.ndarray,
    T: float,
    V0: float,
    kappa: float,
    theta: float,
    rho: float,
    nu: float,
) -> np.ndarray:
    """Classical Heston (1993) characteristic function via closed-form formula."""
    b = kappa - rho * nu * 1j * z_array
    d = np.sqrt(b**2 + nu**2 * (z_array**2 + 1j * z_array))
    g = (b - d) / (b + d)
    exp_dT = np.exp(-d * T)
    C = (kappa * theta / nu**2) * (
        (b - d) * T - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g))
    )
    D_val = (b - d) / nu**2 * (1.0 - exp_dT) / (1.0 - g * exp_dT)
    return np.exp(C + V0 * D_val)


# ===========================================================================
# CGMY module tests
# ===========================================================================


def test_cgmy_exponent_at_zero_is_zero():
    """chi(0) = 0: the CGMY cumulant vanishes at z=0."""
    result = cgmy_exponent(0 + 0j, C_PLUS, C_MINUS, G, M, Y)
    assert abs(result) < 1e-12, f"chi(0) should be ~0, got {result}"


def test_cgmy_exponent_real_z_is_finite():
    """CGMY exponent returns finite values for real z."""
    for z in [0.5, 1.0, 2.0, 5.0]:
        val = cgmy_exponent(float(z), C_PLUS, C_MINUS, G, M, Y)
        assert np.isfinite(np.real(val)) and np.isfinite(np.imag(val)), (
            f"cgmy_exponent({z}) = {val} is not finite"
        )


def test_cgmy_exponent_array_input_shape():
    """cgmy_exponent accepts an array of z and returns the same shape."""
    z_arr = np.array([0.0, 1.0, 2.0, 3.0]) + 0j
    result = cgmy_exponent(z_arr, C_PLUS, C_MINUS, G, M, Y)
    assert result.shape == z_arr.shape, (
        f"Expected shape {z_arr.shape}, got {result.shape}"
    )


def test_cgmy_exponent_array_zero_element():
    """When z array contains 0, that element of chi is 0."""
    z_arr = np.array([0.0, 1.0, 2.0]) + 0j
    result = cgmy_exponent(z_arr, C_PLUS, C_MINUS, G, M, Y)
    assert abs(result[0]) < 1e-12, f"chi(0) element should be ~0, got {result[0]}"


def test_cgmy_drift_compensation_is_real():
    """chi(-i) is real (drift compensation returns a real float)."""
    val = cgmy_drift_compensation(C_PLUS, C_MINUS, G, M, Y)
    assert isinstance(val, float), f"Expected float, got {type(val)}"


def test_cgmy_drift_compensation_is_finite():
    """chi(-i) is finite."""
    val = cgmy_drift_compensation(C_PLUS, C_MINUS, G, M, Y)
    assert np.isfinite(val), f"chi(-i) should be finite, got {val}"


def test_cgmy_drift_compensation_positive():
    """chi(-i) > 0 for CGMY with M > 1 (finite log-MGF)."""
    val = cgmy_drift_compensation(C_PLUS, C_MINUS, G, M, Y)
    assert val > 0, (
        f"chi(-i) = {val:.6f} should be positive for these CGMY parameters"
    )


def test_cpow_real_positive_base():
    """_cpow(4.0, 0.5) = 2.0 (principal-branch square root of 4)."""
    result = _cpow(4.0, 0.5)
    assert abs(result - 2.0) < 1e-12, f"_cpow(4, 0.5) = {result}, expected 2.0"


def test_cpow_principal_branch_sqrt_minus_one():
    """_cpow(-1, 0.5) = i (principal branch: exp(0.5 * i*pi) = i)."""
    result = _cpow(-1 + 0j, 0.5)
    assert abs(result - 1j) < 1e-12, (
        f"_cpow(-1, 0.5) = {result}, expected 1j (principal branch)"
    )


def test_cpow_array_input():
    """_cpow accepts numpy arrays and returns an array."""
    bases = np.array([1.0, 4.0, 9.0])
    result = _cpow(bases, 0.5)
    expected = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(np.real(result), expected, atol=1e-12)


# ===========================================================================
# Rough Heston characteristic function tests
# ===========================================================================


def test_rough_heston_cf_at_zero_is_one():
    """phi_T(0) = 1: CF normalisation at z=0."""
    z0 = np.array([0.0 + 0j])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cf = rough_heston_cf(z0, T=0.01, V0=V0, vol_of_vol=VOL_OF_VOL,
                             kappa=KAPPA, theta=THETA, rho=RHO, H=H,
                             N_steps=50)
    assert abs(cf[0] - 1.0) < 1e-10, f"phi_T(0) = {cf[0]}, expected 1"


def test_rough_heston_cf_modulus_bounded_by_one():
    """|phi_T(u)| <= 1 for real u (characteristic function property)."""
    u_arr = np.array([0.5, 1.0, 2.0, 5.0, 10.0], dtype=complex)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cf = rough_heston_cf(u_arr, T=0.1, V0=V0, vol_of_vol=VOL_OF_VOL,
                             kappa=KAPPA, theta=THETA, rho=RHO, H=H,
                             N_steps=50)
    moduli = np.abs(cf)
    assert np.all(moduli <= 1.0 + 1e-6), (
        f"CF moduli > 1: {moduli[moduli > 1 + 1e-6]}"
    )


def test_rough_heston_cf_output_shape():
    """rough_heston_cf returns array with same length as z_array input."""
    z_arr = np.linspace(0, 5, 8).astype(complex)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cf = rough_heston_cf(z_arr, T=0.05, V0=V0, vol_of_vol=VOL_OF_VOL,
                             kappa=KAPPA, theta=THETA, rho=RHO, H=H,
                             N_steps=20)
    assert cf.shape == z_arr.shape, (
        f"Expected shape {z_arr.shape}, got {cf.shape}"
    )


def test_rough_heston_cf_near_classical_heston():
    """For H=0.499 (near H=0.5), rough Heston CF is close to classical Heston.

    Uses the analytical Heston (1993) formula as reference.
    Relative error should be below 0.1% with N_steps=200.
    """
    kappa_h, theta_h, rho_h, nu_h, V0_h = 2.0, 0.04, -0.5, 0.3, 0.04
    H_near_half = 0.499
    z_arr = np.array([1.0, 2.0, 3.0], dtype=complex)
    T = 1.0

    cf_rough = rough_heston_cf(
        z_arr, T, V0=V0_h, vol_of_vol=nu_h, kappa=kappa_h,
        theta=theta_h, rho=rho_h, H=H_near_half, N_steps=200,
    )
    cf_classical = _heston_cf_analytical(z_arr, T, V0_h, kappa_h, theta_h, rho_h, nu_h)
    rel_err = np.abs(cf_rough - cf_classical) / np.abs(cf_classical)
    assert np.all(rel_err < 1e-3), (
        f"Rough Heston (H=0.499) vs classical Heston: max rel error = {rel_err.max():.2e}"
    )


def test_solve_fractional_riccati_output_shapes():
    """solve_fractional_riccati returns three arrays of shape (N_z,)."""
    z_arr = np.array([0.5, 1.0, 2.0], dtype=complex)
    h_T, I1ma_h_T, int_h = solve_fractional_riccati(
        z_arr, T=0.1, V0=V0, vol_of_vol=VOL_OF_VOL,
        kappa=KAPPA, theta=THETA, rho=RHO, H=H, N_steps=20,
    )
    assert h_T.shape == (3,), f"h_T shape {h_T.shape} != (3,)"
    assert I1ma_h_T.shape == (3,), f"I1ma_h_T shape {I1ma_h_T.shape} != (3,)"
    assert int_h.shape == (3,), f"int_h shape {int_h.shape} != (3,)"


def test_rough_heston_cf_large_N_steps():
    """CF remains stable and finite with large N_steps (edge case)."""
    z_arr = np.array([1.0 + 0j, 2.0 + 0j])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cf = rough_heston_cf(z_arr, T=0.01, V0=V0, vol_of_vol=VOL_OF_VOL,
                             kappa=KAPPA, theta=THETA, rho=RHO, H=H,
                             N_steps=400)
    assert np.all(np.isfinite(np.abs(cf))), f"CF not finite with N_steps=400: {cf}"


# ===========================================================================
# Fourier inversion tests
# ===========================================================================


def test_bs_cf_at_zero_is_one():
    """Pure-Gaussian CF phi_T(0) = 1 (no rough Heston, no jumps)."""
    u = np.array([0.0 + 0j])
    cf = compute_mixed_cf_array(u, T=0.5, rh=None, cgmy=None, sigma_const=0.2)
    assert abs(cf[0] - 1.0) < 1e-15, f"BS phi_T(0) = {cf[0]}, expected 1"


def test_bs_cf_formula_matches_analytical():
    """For sigma_const=0.2, phi_T(u) = exp(-0.04*T*(u^2 + i*u)/2) exactly."""
    sigma = 0.2
    T = 0.5
    u_real = np.array([0.0, 1.0, 2.0, 5.0])
    z_arr = u_real.astype(complex)

    computed = compute_mixed_cf_array(z_arr, T, rh=None, cgmy=None, sigma_const=sigma)
    expected = np.exp(-0.5 * sigma**2 * T * (z_arr**2 + 1j * z_arr))

    np.testing.assert_allclose(computed, expected, atol=1e-14,
                               err_msg="BS CF does not match analytical formula")


def test_mixed_cf_normalization_with_cgmy():
    """Mixed CF (rough Heston + CGMY) at z=0 returns 1."""
    rh = _make_rh()
    cgmy = _make_cgmy()
    z0 = np.array([0.0 + 0j])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cf = compute_mixed_cf_array(z0, T=0.01, rh=rh, cgmy=cgmy, N_steps=20)
    assert abs(cf[0] - 1.0) < 1e-8, f"Mixed CF phi_T(0) = {cf[0]}, expected 1"


def test_compute_p_q_density_bs_iv_close_to_sigma():
    """For pure BS (sigma_const=0.2), ATM IV from Gil-Pelaez ≈ 0.2."""
    sigma = 0.2
    T = 0.1
    p_T, q_T, f0 = compute_p_q_density(T, None, None, sigma_const=sigma, **FAST)
    I = atm_iv(T, p_T, q_T)
    assert abs(I - sigma) < 0.01, (
        f"ATM IV for BS = {I:.4f}, expected close to {sigma}"
    )


def test_compute_p_q_density_returns_finite():
    """compute_p_q_density returns finite p_T, q_T, f0 for typical parameters."""
    rh = _make_rh()
    cgmy = _make_cgmy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_T, q_T, f0 = compute_p_q_density(1e-3, rh, cgmy, **FAST)
    assert np.isfinite(p_T), f"p_T not finite: {p_T}"
    assert np.isfinite(q_T), f"q_T not finite: {q_T}"
    assert np.isfinite(f0), f"f0 not finite: {f0}"


def test_compute_p_q_density_martingale_check():
    """p_T + (1 - q_T) → 1 as T → 0 (ATM call price → 0 under martingale).

    Verified at T=1e-5 using pure BS where the martingale condition holds exactly.
    """
    T = 1e-5
    p_T, q_T, _ = compute_p_q_density(T, None, None, sigma_const=0.2, **FAST)
    martingale_sum = p_T + (1.0 - q_T)
    assert abs(martingale_sum - 1.0) < 0.01, (
        f"p_T + (1 - q_T) = {martingale_sum:.6f}, expected close to 1 for small T"
    )


def test_compute_mixed_cf_array_output_shape():
    """compute_mixed_cf_array returns array of same length as u_array."""
    rh = _make_rh()
    u_arr = np.linspace(0.1, 5.0, 12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cf = compute_mixed_cf_array(u_arr, T=0.01, rh=rh, cgmy=None, N_steps=20)
    assert cf.shape == u_arr.shape, (
        f"Expected shape {u_arr.shape}, got {cf.shape}"
    )


def test_compute_p_q_density_rough_only_finite():
    """compute_p_q_density works without CGMY (rough Heston only)."""
    rh = _make_rh()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_T, q_T, f0 = compute_p_q_density(1e-3, rh, None, **FAST)
    assert np.isfinite(p_T) and np.isfinite(q_T) and np.isfinite(f0), (
        f"p_T={p_T}, q_T={q_T}, f0={f0}"
    )


# ===========================================================================
# ATM quantities tests
# ===========================================================================


def test_atm_iv_bs_equals_sigma():
    """For pure BS, ATM IV equals sigma_const to within quadrature error."""
    sigma = 0.3
    T = 0.25
    p_T, q_T, _ = compute_p_q_density(T, None, None, sigma_const=sigma, **FAST)
    I = atm_iv(T, p_T, q_T)
    assert abs(I - sigma) < 0.015, (
        f"BS ATM IV = {I:.5f}, expected close to sigma = {sigma}"
    )


def test_atm_iv_small_T_approaches_sqrt_V0():
    """ATM IV approaches sqrt(V0) as T decreases.

    For rough Heston + CGMY, |IV(T_small) - sqrt(V0)| < |IV(T_large) - sqrt(V0)|.
    """
    rh = _make_rh()
    cgmy = _make_cgmy()
    sqrt_V0 = np.sqrt(V0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_large, q_large, _ = compute_p_q_density(1e-3, rh, cgmy, **FAST)
        p_small, q_small, _ = compute_p_q_density(1e-4, rh, cgmy, **FAST)

    I_large = atm_iv(1e-3, p_large, q_large)
    I_small = atm_iv(1e-4, p_small, q_small)

    assert abs(I_small - sqrt_V0) < abs(I_large - sqrt_V0), (
        f"Expected |IV(1e-4) - sqrt(V0)| < |IV(1e-3) - sqrt(V0)|, "
        f"got |{I_small:.4f} - {sqrt_V0:.4f}| >= |{I_large:.4f} - {sqrt_V0:.4f}|"
    )


def test_atm_skew_negative_for_negative_rho():
    """ATM skew dI/dk < 0 when rho < 0 (model asymmetry in negative direction)."""
    rh = _make_rh()  # rho = -0.7
    cgmy = _make_cgmy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_T, q_T, _ = compute_p_q_density(1e-3, rh, cgmy, **FAST)
    I = atm_iv(1e-3, p_T, q_T)
    skew = atm_skew(1e-3, p_T, I)
    assert skew < 0, f"Skew = {skew:.4f} should be negative for rho = {RHO}"


def test_atm_skew_grows_as_T_decreases():
    """|ATM skew| increases as T decreases for a rough/jump model."""
    rh = _make_rh()
    cgmy = _make_cgmy()
    T_large, T_small = 1e-3, 1e-4

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_l, q_l, _ = compute_p_q_density(T_large, rh, cgmy, **FAST)
        p_s, q_s, _ = compute_p_q_density(T_small, rh, cgmy, **FAST)

    skew_large = atm_skew(T_large, p_l, atm_iv(T_large, p_l, q_l))
    skew_small = atm_skew(T_small, p_s, atm_iv(T_small, p_s, q_s))

    assert abs(skew_small) > abs(skew_large), (
        f"|skew(T={T_small})| = {abs(skew_small):.4f} should exceed "
        f"|skew(T={T_large})| = {abs(skew_large):.4f}"
    )


def test_theoretical_skew_exponent_rough_dominated():
    """H=0.1, Y=1.5: beta_s = min(H-0.5, (1-Y)/2) = min(-0.4, -0.25) = -0.4."""
    result = theoretical_skew_exponent(H=0.1, Y=1.5)
    assert result == pytest.approx(-0.4), f"Got {result}, expected -0.4"


def test_theoretical_skew_exponent_jump_dominated():
    """H=0.35, Y=1.5: beta_s = min(-0.15, -0.25) = -0.25 (jumps dominate)."""
    result = theoretical_skew_exponent(H=0.35, Y=1.5)
    assert result == pytest.approx(-0.25), f"Got {result}, expected -0.25"


def test_theoretical_skew_exponent_critical():
    """H=0.25, Y=1.5: beta_s = min(-0.25, -0.25) = -0.25 (critical line)."""
    result = theoretical_skew_exponent(H=0.25, Y=1.5)
    assert result == pytest.approx(-0.25), f"Got {result}, expected -0.25"


def test_theoretical_curvature_exponent_rough_dominated():
    """H=0.1, Y=1.5: beta_c = min(2*0.1-1, -1.5/2) = min(-0.8, -0.75) = -0.8."""
    result = theoretical_curvature_exponent(H=0.1, Y=1.5)
    assert result == pytest.approx(-0.8), f"Got {result}, expected -0.8"


def test_theoretical_curvature_exponent_jump_dominated():
    """H=0.2, Y=1.5: beta_c = min(2*0.2-1, -1.5/2) = min(-0.6, -0.75) = -0.75."""
    result = theoretical_curvature_exponent(H=0.2, Y=1.5)
    assert result == pytest.approx(-0.75), f"Got {result}, expected -0.75"


def test_local_log_slope_exact_power_law():
    """local_log_slope exactly recovers beta for a perfect power law qty = c * T^beta."""
    T_arr = np.logspace(-4, -1, 20)
    beta = -0.4
    qty = 1.5 * T_arr**beta
    slopes = local_log_slope(np.log(T_arr), np.log(np.abs(qty)))
    np.testing.assert_allclose(slopes, beta, atol=1e-10,
                               err_msg=f"local_log_slope not recovering beta={beta}")


def test_local_log_slope_output_length():
    """local_log_slope returns N-1 slopes for N inputs."""
    T_arr = np.logspace(-4, -1, 15)
    qty = T_arr**(-0.3)
    slopes = local_log_slope(np.log(T_arr), np.log(qty))
    assert len(slopes) == len(T_arr) - 1, (
        f"Expected {len(T_arr)-1} slopes, got {len(slopes)}"
    )


def test_rough_dominated_skew_coefficient_sign():
    """rough_dominated_skew_coefficient < 0 when rho < 0."""
    coeff = rough_dominated_skew_coefficient(V0=0.04, vol_of_vol=0.45, rho=-0.7, H=0.1)
    assert coeff < 0, f"Rough skew coefficient = {coeff:.4f} should be negative for rho<0"


def test_rough_dominated_skew_coefficient_zero_for_rho_zero():
    """rough_dominated_skew_coefficient = 0 when rho = 0 (no correlation)."""
    coeff = rough_dominated_skew_coefficient(V0=0.04, vol_of_vol=0.45, rho=0.0, H=0.1)
    assert coeff == pytest.approx(0.0, abs=1e-15), (
        f"Rough skew coefficient with rho=0 should be 0, got {coeff}"
    )


def test_jump_dominated_skew_coefficient_negative_asymmetry():
    """jump_dominated_skew_coefficient < 0 when C_minus > C_plus (downward skew)."""
    coeff = jump_dominated_skew_coefficient(V0=0.04, C_plus=0.015, C_minus=0.025, Y=1.5)
    assert coeff < 0, (
        f"Jump skew coefficient = {coeff:.4f} should be negative for C_minus > C_plus"
    )


def test_jump_dominated_skew_coefficient_positive_asymmetry():
    """jump_dominated_skew_coefficient > 0 when C_plus > C_minus (upward skew)."""
    coeff = jump_dominated_skew_coefficient(V0=0.04, C_plus=0.025, C_minus=0.015, Y=1.5)
    assert coeff > 0, (
        f"Jump skew coefficient = {coeff:.4f} should be positive for C_plus > C_minus"
    )


def test_atm_curvature_finite_for_valid_inputs():
    """atm_curvature returns a finite value for valid model outputs."""
    T = 1e-3
    rh = _make_rh()
    cgmy = _make_cgmy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_T, q_T, f0 = compute_p_q_density(T, rh, cgmy, **FAST)
    I = atm_iv(T, p_T, q_T)
    skew = atm_skew(T, p_T, I)
    curv = atm_curvature(T, p_T, q_T, I, skew, f0)
    assert np.isfinite(curv), f"atm_curvature returned non-finite value: {curv}"


# ===========================================================================
# Scenarios tests
# ===========================================================================


def test_make_maturity_grid_sorted():
    """make_maturity_grid returns strictly increasing (sorted) values."""
    T_grid = make_maturity_grid(1e-5, 1e-2, 15)
    diffs = np.diff(T_grid)
    assert np.all(diffs > 0), "Maturity grid is not strictly sorted"


def test_make_maturity_grid_all_positive():
    """All values in the maturity grid are positive."""
    T_grid = make_maturity_grid(1e-8, 1e-3, 20)
    assert np.all(T_grid > 0), f"Non-positive maturities found: {T_grid[T_grid <= 0]}"


def test_make_maturity_grid_correct_size():
    """make_maturity_grid returns an array of the requested length N."""
    for N in [5, 10, 50]:
        T_grid = make_maturity_grid(1e-5, 1e-2, N)
        assert len(T_grid) == N, f"Expected {N} points, got {len(T_grid)}"


def test_make_maturity_grid_endpoints():
    """make_maturity_grid respects T_min and T_max endpoints."""
    T_min, T_max = 1e-6, 1e-2
    T_grid = make_maturity_grid(T_min, T_max, 30)
    assert abs(T_grid[0] - T_min) / T_min < 1e-10, (
        f"First point {T_grid[0]} != T_min {T_min}"
    )
    assert abs(T_grid[-1] - T_max) / T_max < 1e-10, (
        f"Last point {T_grid[-1]} != T_max {T_max}"
    )


def test_skew_scenarios_has_three_elements():
    """SKEW_SCENARIOS contains exactly three scenarios (S1, S2, S3)."""
    assert len(SKEW_SCENARIOS) == 3, (
        f"Expected 3 skew scenarios, got {len(SKEW_SCENARIOS)}"
    )


def test_curvature_scenarios_has_three_elements():
    """CURVATURE_SCENARIOS contains exactly three scenarios (C1, C2, C3)."""
    assert len(CURVATURE_SCENARIOS) == 3, (
        f"Expected 3 curvature scenarios, got {len(CURVATURE_SCENARIOS)}"
    )


def test_skew_scenarios_beta_s_values():
    """Each skew scenario has the correct theoretical beta_s value."""
    expected_betas = {"S1": -0.4, "S2": -0.25, "S3": -0.25}
    for sc in SKEW_SCENARIOS:
        assert sc.beta_s_theory == pytest.approx(expected_betas[sc.name]), (
            f"Scenario {sc.name}: beta_s={sc.beta_s_theory}, "
            f"expected {expected_betas[sc.name]}"
        )


def test_curvature_scenarios_beta_c_values():
    """Curvature scenarios C1 and C2 have the correct theoretical beta_c values."""
    expected_beta_c = {"C1": -0.8, "C2": -0.75, "C3": -0.75}
    for sc in CURVATURE_SCENARIOS:
        if sc.name in expected_beta_c:
            assert sc.beta_c_theory == pytest.approx(expected_beta_c[sc.name]), (
                f"Scenario {sc.name}: beta_c={sc.beta_c_theory}, "
                f"expected {expected_beta_c[sc.name]}"
            )


def test_scenario_s1_is_rough_dominated():
    """Scenario S1 has H < 0.5 - Y/2, so rough term dominates the skew."""
    sc_s1 = SKEW_SCENARIOS[0]
    assert sc_s1.name == "S1"
    beta_rough = sc_s1.rh.H - 0.5
    beta_jump = (1.0 - sc_s1.cgmy.Y) / 2.0
    assert beta_rough < beta_jump, (
        f"S1: beta_rough={beta_rough} should be < beta_jump={beta_jump}"
    )


def test_scenario_c1_has_no_jumps():
    """Scenario C1 is the rough-only model (cgmy is None)."""
    sc_c1 = CURVATURE_SCENARIOS[0]
    assert sc_c1.name == "C1"
    assert sc_c1.cgmy is None, "C1 should have no CGMY component"


# ===========================================================================
# Engine tests
# ===========================================================================

# Small maturity grid for fast engine tests
_T_ENGINE = make_maturity_grid(T_min=1e-4, T_max=1e-3, N=3)


def test_run_scenario_returns_dataframe():
    """run_scenario returns a pandas DataFrame."""
    import pandas as pd

    scenario = SKEW_SCENARIOS[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = run_scenario(scenario, _T_ENGINE, N_steps=20, N_quad=50, verbose=False)
    assert isinstance(df, pd.DataFrame), f"Expected DataFrame, got {type(df)}"


def test_run_scenario_has_expected_columns():
    """run_scenario DataFrame has all required columns."""
    required_cols = {"T", "p_T", "q_T", "f0", "I", "skew", "curvature",
                     "log_T", "beta_s_theory", "beta_c_theory"}
    scenario = SKEW_SCENARIOS[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = run_scenario(scenario, _T_ENGINE, N_steps=20, N_quad=50, verbose=False)
    missing = required_cols - set(df.columns)
    assert not missing, f"DataFrame missing columns: {missing}"


def test_run_scenario_skew_values_negative():
    """For a negatively skewed scenario (rho<0), all finite skew values are negative."""
    scenario = SKEW_SCENARIOS[0]  # S1: rho = -0.7
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = run_scenario(scenario, _T_ENGINE, N_steps=20, N_quad=50, verbose=False)
    valid_skew = df["skew"].dropna()
    assert (valid_skew < 0).all(), (
        f"Not all skew values negative: {valid_skew.values}"
    )


def test_run_scenario_row_count():
    """run_scenario produces one row per maturity in T_grid."""
    scenario = SKEW_SCENARIOS[1]
    T_grid = make_maturity_grid(1e-4, 1e-3, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = run_scenario(scenario, T_grid, N_steps=20, N_quad=50, verbose=False)
    assert len(df) == len(T_grid), (
        f"Expected {len(T_grid)} rows, got {len(df)}"
    )


def test_compute_skew_coefficients_adds_hat_As():
    """compute_skew_coefficients adds hat_A_s, A_s_theory, abs_dev columns."""
    scenario = SKEW_SCENARIOS[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = run_scenario(scenario, _T_ENGINE, N_steps=20, N_quad=50, verbose=False)
    df2 = compute_skew_coefficients(df, scenario)
    for col in ("hat_A_s", "A_s_theory", "abs_dev"):
        assert col in df2.columns, f"Column '{col}' missing from skew coefficients DataFrame"


def test_compute_skew_coefficients_hat_As_finite():
    """hat_A_s values should be finite and non-NaN where skew is valid."""
    scenario = SKEW_SCENARIOS[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = run_scenario(scenario, _T_ENGINE, N_steps=20, N_quad=50, verbose=False)
    df2 = compute_skew_coefficients(df, scenario)
    valid = df2.dropna(subset=["skew"])
    assert (np.isfinite(valid["hat_A_s"])).all(), (
        f"hat_A_s contains non-finite values: {valid['hat_A_s'].values}"
    )


def test_print_summary_runs_without_error(capsys):
    """print_summary executes without raising an exception."""
    scenario = SKEW_SCENARIOS[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = run_scenario(scenario, _T_ENGINE, N_steps=20, N_quad=50, verbose=False)
    print_summary(df, scenario)
    captured = capsys.readouterr()
    assert "S1" in captured.out, "Expected scenario name in print_summary output"
