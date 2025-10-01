import numpy as np
from .params import ModelParams


def step_wealth(Wk: np.ndarray,
                pi_k: np.ndarray,
                c_k: np.ndarray,
                params: ModelParams,
                dt: float,
                z: np.ndarray,
                absorb_at_zero: bool = True) -> np.ndarray:
    """
    Euler–Maruyama step:
    W_{k+1} = W_k + [ r*W_k + pi_k*(mu-r)*W_k - c_k ]*dt
              + pi_k*sigma*W_k*sqrt(dt)*z
    c_k is a *rate* (per year). z ~ N(0,1).
    """
    r, mu, sigma = params.r, params.mu, params.sigma
    drift = (r*Wk + pi_k*(mu - r)*Wk - c_k) * dt
    diff = (pi_k*sigma*Wk) * np.sqrt(dt) * z
    Wn = Wk + drift + diff
    if absorb_at_zero:
        Wn = np.maximum(Wn, 0.0)
    return Wn
