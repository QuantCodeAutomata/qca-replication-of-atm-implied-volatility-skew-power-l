# Project: Rough Heston + CGMY ATM IV Skew Replication

## Structure

```
src/              Core library modules
  cgmy.py         CGMY Lévy cumulant chi(z), drift compensation, _cpow
  rough_heston.py Fractional Riccati solver + rough Heston CF
  fourier_inversion.py  Gil-Pelaez inversion, mixed-model CF, dataclasses
  atm_quantities.py     ATM IV/skew/curvature, theoretical exponents
  scenarios.py    Scenario configs (SKEW_SCENARIOS, CURVATURE_SCENARIOS, make_maturity_grid)
  engine.py       run_scenario, compute_skew_coefficients, print_summary
exp/              Experiment scripts (exp_1..3)
tests/            pytest test suite (79 tests, all passing ~4s)
results/          CSV outputs + plots (generated at runtime)
```

## Running Tests

```bash
# From /workspace/project
python3 -m pytest tests/ -v
```
All 79 tests pass in ~4 seconds with fast params (N_steps=20, N_quad=50).

## Key Mathematical Facts

- **chi(0) = 0** exactly (CGMY exponent at z=0).
- **phi_T(0) = 1** for all model variants (CF normalisation).
- **BS CF**: `exp(-sigma^2*T/2*(z^2+iz))` — matches `compute_mixed_cf_array` exactly when `rh=None, cgmy=None`.
- **Rough Heston H=0.499** converges to classical Heston: relative error < 1e-3 with N_steps=200.
- **Theoretical exponents**: `theoretical_skew_exponent(H,Y) = min(H-0.5, (1-Y)/2)`, `theoretical_curvature_exponent(H,Y) = min(2H-1, -Y/2)`.
- **ATM IV → sqrt(V0)** as T→0 (verified at T=1e-4 for rough Heston + CGMY).
- **Skew is negative** for rho<0 and grows in magnitude as T→0.
- **p_T + (1-q_T) → 1** as T→0 (ATM call price → 0; tested at T=1e-5).

## Important Gotchas

- **H=0.5 exactly**: The fractional Riccati solver degenerates because `(0.0)**0.0 = 1.0` in NumPy makes the I^{1-alpha} weights sum to 0. Do NOT test exact H=0.5 for quantitative CF comparison; use H=0.499.
- **Overflow warnings** are expected and benign at very small T (T < 1e-5) with H < 0.2.
- The `run_exp1/2/3` functions do NOT accept a `verbose` parameter (it is hardcoded in the implementation).
- Exp scripts call `sys.path.insert(0, <project_root>)` at import time; always run pytest from `/workspace/project`.

## Fast Computation Parameters

Use `N_steps=20, N_quad=50` for tests. Production accuracy uses N_steps=100-200, N_quad=200-300.
