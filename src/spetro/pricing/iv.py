import numpy as np
from scipy.stats import norm


def bs_cp(S0: float, K: float, T: float, r: float, sig: float) -> float:
    if T <= 0 or sig <= 0:
        return max(S0 - K * np.exp(-r * T), 0.0)
    rt = np.sqrt(T)
    srt = sig * rt
    d1 = (np.log(S0 / K) + (r + 0.5 * sig * sig) * T) / srt
    d2 = d1 - srt
    return float(S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
