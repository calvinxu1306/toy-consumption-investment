import numpy as np
from src.models.params import ModelParams

def constant_policy(
        wealth_k: np.ndarray,
        _t_k: float,
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

def get_merton_values(params: ModelParams) -> tuple[float, float]:
    """
    Calculates the theoretical optimal controls for Infinite Horizon CRRA.
    Returns: (pi_star, alpha_star)
    """
    mu, r, sigma = params.mu, params.r, params.sigma
    gamma, rho = params.gamma, params.rho

    # 1. Optimal Risky Weight (Merton Ratio)
    # pi* = (mu - r) / (gamma * sigma^2)
    variance = sigma ** 2
    pi_star = (mu - r) / (gamma * variance)

    # 2. Optimal Consumption Rate (MPC)
    # alpha* = (rho - (1-gamma)*(r + 0.5 * gamma * (sigma * pi_star)^2)) / gamma
    # Simplified Merton formula for consumption:
    term_1 = rho / gamma
    term_2 = ((gamma - 1) / gamma) * (r + (mu - r)**2 / (2 * gamma * variance))
    alpha_star = term_1 + term_2

    return pi_star, alpha_star

def merton_policy(
        wealth_k: np.ndarray,
        t_k: float,
        params: ModelParams
) -> tuple[np.ndarray, np.ndarray]:
    """
    Automatically applies the theoretical optimal pi* and alpha*.
    """
    pi_star, alpha_star = get_merton_values(params)
    
    # Delegate to the standard constant policy logic
    return constant_policy(
        wealth_k, t_k, params, 
        pi_const=pi_star, 
        alpha_consume=alpha_star
    )