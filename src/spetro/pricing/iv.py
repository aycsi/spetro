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


def iv_slv(prc: float, K: float, T: float, S0: float, r: float,
           lo: float = 1e-6, hi: float = 5.0, tol: float = 1e-8, mx_it: int = 100) -> float:
    if prc <= 0 or S0 <= 0 or K <= 0 or T <= 0:
        return float('nan')
    lb = bs_cp(S0, K, T, r, lo)
    hb = bs_cp(S0, K, T, r, hi)
    if prc < lb or prc > hb:
        return float('nan')
    for _ in range(mx_it):
        mid = 0.5 * (lo + hi)
        if hi - lo < tol:
            return mid
        cm = bs_cp(S0, K, T, r, mid)
        if abs(cm - prc) < tol:
            return mid
        if cm < prc:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
