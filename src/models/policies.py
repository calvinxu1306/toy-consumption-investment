import numpy as np
from src.models.params import ModelParams


def constant_policy(
        wealth_k: np.ndarray,
        _t_k: float,                      # time not used yet (placeholder)
        params: ModelParams,
        pi_const: float = 0.5,
        alpha_consume: float = 0.04,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Constant policy:
      - pi_k: fixed risky weight (clipped to [0,1])
      - c_k : proportional consumption alpha*W, capped by cbar*W
    """
    pi_k = np.full_like(wealth_k, pi_const, dtype=float)
    pi_k = np.clip(pi_k, 0.0, 1.0)

    c_rate = alpha_consume * wealth_k
    c_cap = params.cbar * wealth_k
    c_k = np.minimum(c_rate, c_cap)
    c_k = np.where(wealth_k <= 0.0, 0.0, c_k)
    return pi_k, c_k


def zero_consumption_with_constant_pi(
        wealth_k: np.ndarray,
        _t_k: float,
        _params: ModelParams,
        pi_const: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Sanity policy: invest with constant pi, consume nothing."""
    pi_k = np.full_like(wealth_k, pi_const, dtype=float)
    pi_k = np.clip(pi_k, 0.0, 1.0)
    c_k = np.zeros_like(wealth_k, dtype=float)
    return pi_k, c_k
