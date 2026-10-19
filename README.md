# Replication of ATM Implied-Volatility Skew Power-Law Exponents in Mixed Rough Heston + CGMY Models

Numerical replication of short-maturity asymptotic results for the at-the-money (ATM)
implied-volatility skew (and curvature) in a mixed model combining rough Heston stochastic
volatility and infinite-variation CGMY jumps.

Three experiments are implemented:

| Exp | Title | Key Result |
|-----|-------|-----------|
| 1 | ATM skew power-law exponents | β_s → min(H−½, (1−Y)/2) as T→0 |
| 2 | ATM skew leading coefficients | Â_s → A_s as T→0 |
| 3 | Skew–curvature joint exponents | Mixed models break pure-model β_c = 2β_s / β_c = β_s−½ relations |

## Repository Structure

```
.
├── data/                  # (empty — synthetic model inputs only, no market data)
├── src/
│   ├── cgmy.py            # CGMY characteristic exponent χ(z)
│   ├── rough_heston.py    # Rough Heston CF via fractional Riccati equation
│   ├── fourier_inversion.py  # Gil-Pelaez p_T, q_T, density inversion
│   ├── atm_quantities.py  # ATM IV, skew, curvature + asymptotic formulas
│   ├── scenarios.py       # Experiment scenario configurations (S1–S3, C1–C3)
│   └── engine.py          # Main compute loop: run_scenario()
├── exp/
│   ├── exp_1_skew_power.py        # Exp 1 runner
│   ├── exp_2_skew_coefficients.py # Exp 2 runner
│   └── exp_3_skew_curvature.py    # Exp 3 runner
├── tests/
│   ├── test_models.py      # 79 unit tests for all modules
│   └── test_exp_scripts.py # Smoke-tests for experiment runners
├── results/
│   ├── RESULTS.md
│   ├── exp1_S{1,2,3}.csv, exp1_skew_power.png
│   ├── exp2_S{1,2,3}.csv, exp2_skew_coefficients.png
│   └── exp3_C{1,2,3}.csv, exp3_skew_curvature.png
└── README.md
```

## The Model

The risk-neutral log-price follows the mixed process:

```
X_t = X0 − η t − ½ ∫₀ᵗ Vₛ ds + ∫₀ᵗ √Vₛ (ρ dWₛ + √(1−ρ²) dBₛ) + Lₜ
```

with rough Heston variance:

```
Vₜ = V0 + κ ∫₀ᵗ K(t−s)(θ − Vₛ) ds + ν ∫₀ᵗ K(t−s) √Vₛ dWₛ
K(t) = t^{H−½} / Γ(H+½)
```

and compensated CGMY jumps with characteristic exponent:

```
χ(z) = Γ(−Y) [C₊ ((M−iz)^Y − M^Y) + C₋ ((G+iz)^Y − G^Y)]
```

The characteristic function factorises as:

```
φ_T(z) = φ_T^cont(z) × exp(T [χ(z) − iz χ(−i)])
```

where `φ_T^cont` is computed by solving the fractional Riccati equation via the Adams-Bashforth
rectangle scheme (El Euch & Rosenbaum 2019).

## ATM Quantities

Gil-Pelaez inversion gives:

```
p_T = ½ + (1/π) ∫₀^∞ Im(φ_T(u))/u du
q_T = ½ + (1/π) ∫₀^∞ Im(φ_T(u−i))/u du
```

Then closed-form:

```
I(T) = (2/√T) Φ⁻¹((1 + q_T − p_T)/2)
∂_k I = [Φ(−I√T/2) − p_T] / [√T φ(I√T/2)]
∂²_k I = [∂²_k C|model − ∂²_k C^BS − 2 vanna·skew − volga·skew²] / vega
```

## Installation

```bash
pip install numpy scipy pandas matplotlib pytest
```

All required packages are in the scientific Python stack (no special installs needed).

## Running the Experiments

```bash
# Run all experiments (saves CSVs and PNGs to results/)
python exp/exp_1_skew_power.py
python exp/exp_2_skew_coefficients.py
python exp/exp_3_skew_curvature.py
```

## Running Tests

```bash
pytest tests/ -v
```

All 79 tests pass on Python 3.12.

## Key Results

| Scenario | β_s theory | β_s estimated | Error |
|----------|-----------|---------------|-------|
| S1 (rough-dominated) | −0.400 | −0.4024 | 0.6% |
| S2 (jump-dominated)  | −0.250 | −0.2551 | 2.0% |
| S3 (critical line)   | −0.250 | −0.2553 | 2.1% |

See `results/RESULTS.md` for full numerical tables and discussion.

## Implementation Notes

- **Fractional Riccati**: Custom Adams-Bashforth rectangle scheme. No mainstream Python package
  provides this for rough Heston + CGMY (confirmed via Context7).
- **Quadrature**: Vectorised trapezoidal rule on a log-spaced grid with
  `U_max = 20/√(V₀ T)`, verified numerically across T ∈ [1e−11, 1e−3].
- **Complex powers**: Principal branch throughout (`numpy.log` default).
- **Precision**: float64 is sufficient for T ≥ 1e−8; below that, the skew
  inversion remains stable but curvature estimates show pre-asymptotic behaviour.
