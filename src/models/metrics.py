import numpy as np

from src.models.dynamics import WEALTH_FLOOR
from src.models.params import ModelParams


def summarize_terminal_wealth(terminal_wealth: np.ndarray) -> dict[str, float]:
    """Basic stats for W_T."""
    return {
        "mean": float(np.mean(terminal_wealth)),
        "median": float(np.median(terminal_wealth)),
        "p05": float(np.quantile(terminal_wealth, 0.05)),
        "p95": float(np.quantile(terminal_wealth, 0.95)),
        "prob_ruin": float(np.mean(terminal_wealth <= WEALTH_FLOOR)),
    }


def certainty_equivalent(du: float, params: ModelParams, times: np.ndarray) -> float:
    """
    Constant consumption rate whose discounted utility equals `du`: the sure
    stream the agent would accept in place of the policy.

    Utility levels have no natural scale (with gamma = 2, u(c) = -1/c), so
    "% more utility" means little. This is in consumption units instead, and
    because CRRA utility is homogeneous, CE_a / CE_b - 1 is the extra
    consumption policy b would need to match policy a. With kappa > 0 the
    bequest is valued at the same level.
    """
    dt = times[1] - times[0]
    weight = (np.sum(np.exp(-params.rho * times[:-1])) * dt
              + params.kappa * np.exp(-params.rho * params.T))
    if abs(params.gamma - 1.0) < 1e-12:
        return float(np.exp(du / weight))
    return float(((1 - params.gamma) * du / weight) ** (1 / (1 - params.gamma)))
