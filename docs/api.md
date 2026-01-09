# API Reference

## Engine

```python
from spetro import RoughVolatilityEngine
e = RoughVolatilityEngine(backend="jax", device=None, precision="float32")
```

### simulate(model, n_paths, n_steps, T, S0=100.0, key=None)
simulate paths

**params:**
- `model`: RoughVolatilityModel
- `n_paths`: int, paths to simulate
- `n_steps`: int, time steps
- `T`: float, maturity
- `S0`: float, initial price
- `key`: random key/seed

**returns:** tuple (S, V)

### price(model, payoff_fn, n_paths, n_steps, T, S0=100.0, key=None, antithetic=True)
price option

**params:**
- `model`: RoughVolatilityModel
- `payoff_fn`: function(S) -> payoff
- `n_paths`: int
- `n_steps`: int
- `T`: float
- `S0`: float
- `key`: random key/seed
- `antithetic`: bool

**returns:** dict {price, std_error, paths}

### greeks(model, payoff_fn, n_paths, n_steps, T, S0=100.0, key=None)
calc greeks

**params:**
- `model`: RoughVolatilityModel
- `payoff_fn`: function(S) -> payoff
- `n_paths`: int
- `n_steps`: int
- `T`: float
- `S0`: float
- `key`: random key/seed

**returns:** dict {price, delta, gamma}

## Models

### RoughBergomi
```python
from spetro import RoughBergomi
model = RoughBergomi(H=0.07, eta=1.9, rho=-0.9, xi=0.235**2, r=0.0)
```

**params:**
- `H`: float, hurst parameter (0 < H < 0.5)
- `eta`: float, vol of vol
- `rho`: float, correlation
- `xi`: float, initial variance
- `r`: float, risk-free rate

### RoughHeston
```python
from spetro import RoughHeston
model = RoughHeston(H=0.07, nu=0.3, theta=0.02, rho=-0.7, V0=0.02, r=0.0)
```

**params:**
- `H`: float, hurst parameter (0 < H < 0.5)
- `nu`: float, vol of vol
- `theta`: float, mean reversion
- `rho`: float, correlation
- `V0`: float, initial variance
- `r`: float, risk-free rate

## Pricer

```python
from spetro import Pricer
p = Pricer(engine)
```

### price_european(model, option_type, K, T, S0=100.0, n_paths=100000, n_steps=252, antithetic=True)
price european option

**params:**
- `model`: RoughVolatilityModel
- `option_type`: str, "call" or "put"
- `K`: float, strike
- `T`: float, maturity
- `S0`: float, initial price
- `n_paths`: int
- `n_steps`: int
- `antithetic`: bool

**returns:** dict {price, std_error, paths}

### price_asian(model, option_type, K, T, S0=100.0, n_paths=100000, n_steps=252)
price asian option

**params:**
- `model`: RoughVolatilityModel
- `option_type`: str, "call"
- `K`: float, strike
- `T`: float, maturity
- `S0`: float, initial price
- `n_paths`: int
- `n_steps`: int

**returns:** dict {price, std_error, paths}

### price_barrier(model, K, barrier, barrier_type, T, S0=100.0, n_paths=100000, n_steps=252)
price barrier option

**params:**
- `model`: RoughVolatilityModel
- `K`: float, strike
- `barrier`: float, barrier level
- `barrier_type`: str, "up_and_out" or "down_and_out"
- `T`: float, maturity
- `S0`: float, initial price
- `n_paths`: int
- `n_steps`: int

**returns:** dict {price, std_error, paths}

### price_custom(model, payoff_fn, T, S0=100.0, n_paths=100000, n_steps=252, antithetic=True)
price custom option

**params:**
- `model`: RoughVolatilityModel
- `payoff_fn`: function(S) -> payoff
- `T`: float, maturity
- `S0`: float, initial price
- `n_paths`: int
- `n_steps`: int
- `antithetic`: bool

**returns:** dict {price, std_error, paths}

### greeks(model, option_type, K, T, S0=100.0, n_paths=100000, n_steps=252)
calc greeks

**params:**
- `model`: RoughVolatilityModel
- `option_type`: str, "call" or "put"
- `K`: float, strike
- `T`: float, maturity
- `S0`: float, initial price
- `n_paths`: int
- `n_steps`: int

**returns:** dict {price, delta, gamma}

## MonteCarloPricer

```python
from spetro import MonteCarloPricer
mp = MonteCarloPricer(engine)
```

### price_portfolio(models, payoff_fns, weights, T, S0, n_paths=100000, n_steps=252, correlations=None)
price portfolio

**params:**
- `models`: list of RoughVolatilityModel
- `payoff_fns`: list of payoff functions
- `weights`: list of float, portfolio weights
- `T`: float, maturity
- `S0`: list of float, initial prices
- `n_paths`: int
- `n_steps`: int
- `correlations`: numpy array, correlation matrix

**returns:** dict {portfolio_value, individual_prices, weights}

### variance_reduction_price(model, payoff_fn, control_variate_fn, T, S0=100.0, n_paths=100000, n_steps=252, beta=None)
price with control variate

**params:**
- `model`: RoughVolatilityModel
- `payoff_fn`: function(S) -> payoff
- `control_variate_fn`: function(S) -> control payoff
- `T`: float, maturity
- `S0`: float, initial price
- `n_paths`: int
- `n_steps`: int
- `beta`: float, control variate coefficient

**returns:** dict {price, std_error, variance_reduction, paths}

## Calibrator

```python
from spetro import Calibrator
c = Calibrator(engine)
```

### calibrate_to_surface(model_class, market_prices, S0=100.0, initial_params=None, bounds=None, optimizer="adam", max_iter=1000, tolerance=1e-6)
calibrate model to market

**params:**
- `model_class`: model class (RoughBergomi, RoughHeston)
- `market_prices`: dict {(K, T): price}
- `S0`: float, initial price
- `initial_params`: dict, initial parameters
- `bounds`: dict, parameter bounds
- `optimizer`: str, "adam"
- `max_iter`: int, max iterations
- `tolerance`: float, convergence tolerance

**returns:** dict {model, parameters, objective_value, iterations, success}

### validate_calibration(model, market_prices, S0=100.0)
validate calibration

**params:**
- `model`: RoughVolatilityModel
- `market_prices`: dict {(K, T): price}
- `S0`: float, initial price

**returns:** dict {individual_results, mean_absolute_error, max_relative_error}

## NeuralSurrogate

```python
from spetro import NeuralSurrogate
ns = NeuralSurrogate(engine, backend="jax")
```

### generate_training_data(model, param_ranges, option_configs, n_samples=10000, n_paths=50000)
generate training data

**params:**
- `model`: RoughVolatilityModel
- `param_ranges`: dict {param: (low, high)}
- `option_configs`: list of dicts {K, T, S0, option_type}
- `n_samples`: int, number of samples
- `n_paths`: int, paths per sample

**returns:** tuple (X, y)

### train(X, y, validation_split=0.2, epochs=1000, learning_rate=1e-3, batch_size=512)
train neural network

**params:**
- `X`: features array
- `y`: targets array
- `validation_split`: float, validation fraction
- `epochs`: int, training epochs
- `learning_rate`: float, learning rate
- `batch_size`: int, batch size

**returns:** dict {train_loss, val_loss}

### predict(features)
predict price

**params:**
- `features`: list or array, input features

**returns:** float, predicted price

## Payoffs

### european_call(K)
european call payoff

**params:**
- `K`: float, strike

**returns:** function(S) -> payoff

### european_put(K)
european put payoff

**params:**
- `K`: float, strike

**returns:** function(S) -> payoff

### asian_call(K)
asian call payoff

**params:**
- `K`: float, strike

**returns:** function(S) -> payoff

### barrier_call(K, barrier, barrier_type="up_and_out")
barrier call payoff

**params:**
- `K`: float, strike
- `barrier`: float, barrier level
- `barrier_type`: str, "up_and_out" or "down_and_out"

**returns:** function(S) -> payoff

### basket_call(weights, K)
basket call payoff

**params:**
- `weights`: list, asset weights
- `K`: float, strike

**returns:** function(S) -> payoff

## Backends

### JAX
```python
backend = JAXBackend(device=None, precision="float32")
```

### Torch
```python
backend = TorchBackend(device=None, precision="float32")
