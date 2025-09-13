from typing import Any, Optional, Tuple, Dict
from abc import ABC, abstractmethod
import numpy as np

from .backends import Backend


class RoughVolatilityModel(ABC):
    @abstractmethod
    def simulate(
        self,
        backend: Backend,
        n_paths: int,
        n_steps: int,
        T: float,
        S0: float,
        key: Optional[Any] = None
    ) -> Tuple[Any, Any]:
        pass


class RoughBergomi(RoughVolatilityModel):
    def __init__(
        self,
        H: float = 0.07,
        eta: float = 1.9,
        rho: float = -0.9,
        xi: float = 0.235**2,
        r: float = 0.0
    ):
        self.H = H
        self.eta = eta
        self.rho = rho
        self.xi = xi
        self.r = r
        
        if not (0 < H < 0.5):
            raise ValueError("hurst parameter must be in (0, 0.5)")
    
    def simulate(
        self,
        backend: Backend,
        n_paths: int,
        n_steps: int,
        T: float,
        S0: float,
        key: Optional[Any] = None
    ) -> Tuple[Any, Any]:
        dt = T / n_steps
        
        if key is None:
            if hasattr(backend, 'random') and hasattr(backend.random, 'PRNGKey'):
                key = backend.random.PRNGKey(42)
            else:
                key = 42
        
        if hasattr(backend, 'random') and hasattr(backend.random, 'split'):
            k1, k2 = backend.random.split(key)
        else:
            k1, k2 = key, key + 1
        
        dW1 = backend.random_normal(k1, (n_paths, n_steps)) * backend.sqrt(dt)
        dW2 = backend.random_normal(k2, (n_paths, n_steps)) * backend.sqrt(dt)
        
        dB = self.rho * dW1 + backend.sqrt(1 - self.rho**2) * dW2
        
        t_grid = backend.array([i * dt for i in range(n_steps + 1)])
        
        Y = self._fractional_brownian_motion(backend, dW1, t_grid, self.H)
        
        V = backend.zeros((n_steps + 1, n_paths))
        if hasattr(V, 'at'):
            V = V.at[0].set(self.xi)
            for i in range(n_steps):
                vol_term = self.xi * backend.exp(self.eta * Y[i] - 0.5 * self.eta**2 * t_grid[i+1])
                V = V.at[i+1].set(vol_term)
        else:
            V[0] = self.xi
            for i in range(n_steps):
                vol_term = self.xi * backend.exp(self.eta * Y[i] - 0.5 * self.eta**2 * t_grid[i+1])
                V[i+1] = vol_term
        
        log_S = backend.zeros((n_paths, n_steps + 1))
        if hasattr(log_S, 'at'):
            log_S = log_S.at[:, 0].set(np.log(S0))
        else:
            log_S[:, 0] = np.log(S0)
        
        for i in range(n_steps):
            vol = backend.sqrt(V[i])
            drift = (self.r - 0.5 * V[i]) * dt
            diffusion = vol * dB[:, i]
            
            if hasattr(log_S, 'at'):
                log_S = log_S.at[:, i + 1].set(log_S[:, i] + drift + diffusion)
            else:
                log_S[:, i + 1] = log_S[:, i] + drift + diffusion
        
        S = backend.exp(log_S)
        
        return S, V
    
    def _fractional_brownian_motion(
        self, 
        backend: Backend, 
        dW: Any, 
        t_grid: Any, 
        H: float
    ) -> Any:
        n_paths, n_steps = dW.shape
        dt = t_grid[1] - t_grid[0]
        
        g_kernel = self._riemann_liouville_kernel(backend, t_grid[1:], H)
        
        Y = backend.zeros((n_steps, n_paths))
        
        for i in range(n_steps):
            if i == 0:
                weights = backend.array([g_kernel[0]])
                dW_slice = dW[:, :1]
            else:
                weights = g_kernel[:i+1]
                if hasattr(backend, 'jnp'):
                    weights = weights[::-1]
                else:
                    weights = backend.torch.flip(weights, dims=[0])
                dW_slice = dW[:, :i+1]
            
            if hasattr(backend, 'jnp'):
                Y = Y.at[i].set(backend.jnp.dot(weights, dW_slice.T))
            else:
                Y[i] = backend.torch.matmul(weights.unsqueeze(0), dW_slice.T).squeeze()
        
        return Y
    
    def _riemann_liouville_kernel(self, backend: Backend, t: Any, H: float) -> Any:
        alpha = H + 0.5
        
        def gamma_func(x):
            if hasattr(backend, 'jax'):
                from jax.scipy.special import gamma
                return gamma(x)
            else:
                return backend.torch.exp(backend.torch.lgamma(backend.array(x)))
        
        normalization = backend.sqrt(2 * H * gamma_func(1.5 - H) / gamma_func(H + 0.5))
        
        kernel = normalization * (t ** (H - 0.5))
        
        return kernel


class RoughHeston(RoughVolatilityModel):
    def __init__(
        self,
        H: float = 0.07,
        nu: float = 0.3,
        theta: float = 0.02,
        rho: float = -0.7,
        V0: float = 0.02,
        r: float = 0.0
    ):
        self.H = H
        self.nu = nu
        self.theta = theta
        self.rho = rho
        self.V0 = V0
        self.r = r
        
        if not (0 < H < 0.5):
            raise ValueError("hurst parameter must be in (0, 0.5)")
    
    def simulate(
        self,
        backend: Backend,
        n_paths: int,
        n_steps: int,
        T: float,
        S0: float,
        key: Optional[Any] = None
    ) -> Tuple[Any, Any]:
        dt = T / n_steps
        
        if key is None:
            if hasattr(backend, 'random') and hasattr(backend.random, 'PRNGKey'):
                key = backend.random.PRNGKey(42)
            else:
                key = 42
        
        if hasattr(backend, 'random') and hasattr(backend.random, 'split'):
            keys = backend.random.split(key, 3)
            k1, k2, k3 = keys[0], keys[1], keys[2]
        else:
            k1, k2, k3 = key, key + 1, key + 2
        
        dW1 = backend.random_normal(k1, (n_paths, n_steps)) * backend.sqrt(dt)
        dW2 = backend.random_normal(k2, (n_paths, n_steps)) * backend.sqrt(dt)
        dZ = backend.random_normal(k3, (n_paths, n_steps)) * backend.sqrt(dt)
        
        dB = self.rho * dW1 + backend.sqrt(1 - self.rho**2) * dW2
        
        V = backend.zeros((n_paths, n_steps + 1))
        S = backend.zeros((n_paths, n_steps + 1))
        
        if hasattr(V, 'at'):
            V = V.at[:, 0].set(self.V0)
            S = S.at[:, 0].set(S0)
        else:
            V[:, 0] = self.V0
            S[:, 0] = S0
        
        for i in range(n_steps):
            vol_of_vol = self.nu * (V[:, i] ** 0.5)
            
            dV = self.theta * dt + vol_of_vol * dZ[:, i]
            
            if hasattr(V, 'at'):
                V = V.at[:, i + 1].set(backend.jnp.maximum(V[:, i] + dV, 0.0))
            else:
                V[:, i + 1] = backend.torch.clamp(V[:, i] + dV, min=0.0)
            
            drift = self.r * dt
            diffusion = backend.sqrt(V[:, i]) * dB[:, i]
            
            if hasattr(S, 'at'):
                S = S.at[:, i + 1].set(S[:, i] * backend.exp(drift - 0.5 * V[:, i] * dt + diffusion))
            else:
                S[:, i + 1] = S[:, i] * backend.exp(drift - 0.5 * V[:, i] * dt + diffusion)
        
        return S, V
