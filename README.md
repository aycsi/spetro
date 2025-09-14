# Spetro

[![PyPI](https://img.shields.io/pypi/v/spetro.svg)](https://pypi.org/project/spetro/)
[![thoughts](https://img.shields.io/badge/docs-thoughts-blue)](docs/thoughts.md)
![Beta](https://img.shields.io/badge/status-beta-orange)
![Beta](https://img.shields.io/badge/beta-orange?style=flat)
![Development Status](https://img.shields.io/badge/dev--status-beta-yellow)
[![api](https://img.shields.io/badge/docs-api-purple)](docs/api.md)
[![Downloads](https://pepy.tech/badge/spetro)](https://pepy.tech/project/spetro)

Spetro implements rough volatility models for derivatives with accelerated  simulation and automatic differentiation.

## Frameworks

### Rough Bergomi Model
The rough Bergomi model extends classical stochastic volatility with fractional Brownian motion:

$$dS(t) = r S(t) dt + \sqrt{V(t)} S(t) dB(t)$$

$$V(t) = \xi \exp\left(\eta Y(t) - \frac{1}{2}\eta^2 t\right)$$

Where $Y(t)$ is fractional Brownian motion with Hurst parameter $H \in (0, 0.5)$ constructed via Riemann-Liouville kernel:

$$g(t) = \sqrt{\frac{2H \Gamma(3/2-H)}{\Gamma(H+1/2)}} t^{H-1/2}$$

### Rough Heston Model  
Fractional extension of Heston with mean-reverting volatility:

$$dS(t) = r S(t) dt + \sqrt{V(t)} S(t) dB(t)$$

$$dV(t) = \theta dt + \nu \sqrt{V(t)} dZ(t)$$

With correlation $\rho$ between $dB(t)$ and $dZ(t)$, and $V(0) = V_0$.


## Installation

```bash
pip install spetro
```

### Testing
![performance](docs/testings-figure-b.png)

### Architecture  
![schema](docs/schema-thoughts-figure-a.png)

