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
      - pi_k: fixed risky weight (clipped to [0, pi_max])
      - c_k : proportional consumption alpha*W
    """
    pi_k = np.full_like(wealth_k, pi_const, dtype=float)
    pi_k = np.clip(pi_k, 0.0, params.pi_max)

    c_k = alpha_consume * wealth_k
    c_k = np.where(wealth_k <= 0.0, 0.0, c_k)
    return pi_k, c_k

def zero_consumption_with_constant_pi(
        wealth_k: np.ndarray,
        _t_k: float,
        params: ModelParams,
        pi_const: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Sanity policy: invest with constant pi, consume nothing."""
    pi_k = np.full_like(wealth_k, pi_const, dtype=float)
    pi_k = np.clip(pi_k, 0.0, params.pi_max)
    c_k = np.zeros_like(wealth_k, dtype=float)
    return pi_k, c_k

def get_merton_values(params: ModelParams) -> tuple[float, float]:
    """
    Calculates the theoretical optimal controls for Infinite Horizon CRRA,
    with the risky weight restricted to [0, pi_max].
    Returns: (pi_star, alpha_star)

    alpha_star <= 0 means the infinite-horizon problem has no solution
    (possible only for gamma < 1); the finite-horizon policy is still valid.
    """
    mu, r, sigma = params.mu, params.r, params.sigma
    gamma, rho = params.gamma, params.rho

    # 1. Optimal Risky Weight (Merton Ratio)
    # pi* = (mu - r) / (gamma * sigma^2), clipped to the allowed range.
    # The objective is concave in pi, so clipping gives the constrained optimum.
    if sigma > 0:
        pi_unconstrained = (mu - r) / (gamma * sigma ** 2)
    else:
        pi_unconstrained = np.inf if mu > r else 0.0
    pi_star = float(np.clip(pi_unconstrained, 0.0, params.pi_max))

    # 2. Optimal Consumption Rate (MPC)
    # alpha* = (rho - (1-gamma) * R) / gamma, where R is the certainty-equivalent
    # return of the portfolio. At the unconstrained pi*, R = r + (mu-r)^2 / (2*gamma*sigma^2).
    ce_return = r + pi_star * (mu - r) - 0.5 * gamma * (sigma * pi_star) ** 2
    alpha_star = (rho - (1 - gamma) * ce_return) / gamma

    return pi_star, alpha_star

def merton_policy(
        wealth_k: np.ndarray,
        t_k: float,
        params: ModelParams
) -> tuple[np.ndarray, np.ndarray]:
    """
    Static baseline: applies the infinite-horizon pi* and alpha* at every date,
    ignoring how much time is left.
    """
    pi_star, alpha_star = get_merton_values(params)
    
    # Delegate to the standard constant policy logic
    return constant_policy(
        wealth_k, t_k, params, 
        pi_const=pi_star, 
        alpha_consume=max(alpha_star, 0.0)
    )

def finite_horizon_merton_policy(
        wealth_k: np.ndarray,
        t_k: float,
        params: ModelParams
) -> tuple[np.ndarray, np.ndarray]:
    """
    Closed-form optimum of the finite-horizon problem (continuous time).

    The risky weight stays at pi*; the consumption rate rises as T approaches:
        alpha(t) = 1 / g(t),   tau = T - t
        g(t) = (1 - exp(-alpha* tau)) / alpha* + kappa^(1/gamma) * exp(-alpha* tau)
    With kappa = 0 this is alpha* / (1 - exp(-alpha* tau)) ~ 1/tau near T,
    so wealth is run down to zero by the horizon.
    """
    pi_star, alpha_star = get_merton_values(params)
    tau = max(params.T - t_k, 0.0)

    # (1 - e^{-alpha* tau}) / alpha*, which tends to tau as alpha* -> 0
    if abs(alpha_star) < 1e-12:
        annuity = tau
    else:
        annuity = -np.expm1(-alpha_star * tau) / alpha_star
    g = annuity + params.kappa ** (1 / params.gamma) * np.exp(-alpha_star * tau)
    alpha = 1.0 / g if g > 0 else np.inf

    return constant_policy(
        wealth_k, t_k, params,
        pi_const=pi_star,
        alpha_consume=alpha
    )
