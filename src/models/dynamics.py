import numpy as np
from src.models.params import ModelParams

def step_wealth(
        wealth_k: np.ndarray,
        pi_k: np.ndarray,
        c_k: np.ndarray,
        params: ModelParams,
        dt: float,
        z: np.ndarray,
        absorb_at_zero: bool = True,
) -> np.ndarray:
    """
    Euler–Maruyama:
      W_{k+1} = W_k + [ r*W_k + pi_k*(mu-r)*W_k - c_k ]*dt
                + pi_k*sigma*W_k*sqrt(dt)*z
    c_k is a *rate* (per year). z ~ N(0,1).
    """
    r, mu, sigma = params.r, params.mu, params.sigma
    drift = (r * wealth_k + pi_k * (mu - r) * wealth_k - c_k) * dt
    diff = (pi_k * sigma * wealth_k) * np.sqrt(dt) * z
    wealth_next = wealth_k + drift + diff
    if absorb_at_zero:
        wealth_next = np.maximum(wealth_next, 0.0)
    return wealth_next
