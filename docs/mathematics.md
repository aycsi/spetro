# Mathematics

## RoughBergomi

The RoughBergomi model simulates asset price S_t and variance V_t:

```
dS_t = rS_t dt + √V_t S_t dB_t
V_t = ξ exp(η Y_t^H - 0.5 η² t^(2H))
```

**Parameters:**
- H: Hurst parameter (0 < H < 0.5)
- η: Volatility of volatility
- ρ: Correlation between price and volatility
- ξ: Initial variance
- r: Risk-free rate

**Implementation:**
- Fractional Brownian motion Y_t^H generated via Riemann-Liouville kernel
- Variance V_t computed as exponential of fractional Brownian motion
- Price S_t simulated using Euler scheme with correlated Brownian motion

## RoughHeston

The RoughHeston model extends Heston with rough volatility:

```
dS_t = rS_t dt + √V_t S_t dB_t
dV_t = θ(V₀ - V_t)dt + ν√(2H) Y_t^H √dt
```

**Parameters:**
- H: Hurst parameter (0 < H < 0.5)
- ν: Volatility of variance
- θ: Mean reversion speed
- V₀: Long-term variance level
- ρ: Correlation between price and volatility
- r: Risk-free rate

**Implementation:**
- Variance process uses mean reversion with rough volatility term
- Variance clipping: `max(v, 1e-8)` prevents numerical instability
- Price simulation uses exponential form with variance-dependent drift

## Fractional Brownian Motion

**Riemann-Liouville Kernel:**
```
K_H(t,s) = √(2H Γ(1.5-H) / Γ(H+0.5)) × (t-s)^(H-0.5)
```

**Implementation:**
- Kernel computed using gamma functions
- Convolution with Gaussian increments: `Y_i = Σ K_H(t_i, t_j) ΔW_j`
- JAX: `jnp.convolve` with reversed kernel
- PyTorch: `torch.conv1d` with padding

## Neural Surrogate

**Architecture:**
- Multi-layer perceptron: [64, 128, 64, 32] neurons
- ReLU activations between layers
- Single output for option price

**Training:**
- Adam optimizer, learning rate 1e-3
- Batch size 512
- Validation split 20%
- MSE loss function

**Data Generation:**
- Parameter ranges: H ∈ [0.05, 0.15], η ∈ [1.0, 3.0], ρ ∈ [-0.95, -0.5], ξ ∈ [0.01, 0.1]
- Option configurations: strikes 90-110, maturities 0.1-1.0 years
- Labels generated via Monte Carlo simulation

## Monte Carlo

**Error Control:**
- Standard error: σ/√N where N = number of paths
- 95% confidence interval: P̂ ± 1.96 σ/√N
- Antithetic variates for variance reduction

**Implementation:**
- Path-independent simulation enables parallelization
- Backend-specific random number generation
- Automatic differentiation for Greeks calculation

## References

1. Gatheral, Jaisson, Rosenbaum (2018)
2. Bayer, Friz, Gatheral (2016)
3. El Euch, Rosenbaum (2019)