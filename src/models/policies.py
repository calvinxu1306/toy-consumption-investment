import numpy as np
from .params import ModelParams


def constant_policy(Wk: np.ndarray,
                    tk: float,
                    params: ModelParams,
                    pi_const: float = 0.5,
                    alpha_consume: float = 0.04) -> tuple[np.ndarray, np.ndarray]:
    """
    pi_k: constant risky weight in [0,1]
    c_k: proportional consumption rate alpha*Wk, capped by cbar*Wk
    """
    pi_k = np.full_like(Wk, pi_const, dtype=float)
    pi_k = np.clip(pi_k, 0.0, 1.0)

    c_rate = alpha_consume * Wk
    c_cap  = params.cbar * Wk
    c_k = np.minimum(c_rate, c_cap)
    c_k = np.where(Wk <= 0.0, 0.0, c_k)
    return pi_k, c_k

def zero_consumption_with_constant_pi(Wk: np.ndarray,
                                      tk: float,
                                      params: ModelParams,
                                      pi_const: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    """Sanity check policy: invest with constant pi, consume nothing."""
    pi_k = np.full_like(Wk, pi_const, dtype=float)
    pi_k = np.clip(pi_k, 0.0, 1.0)
    c_k = np.zeros_like(Wk, dtype=float)
    return pi_k, c_k
