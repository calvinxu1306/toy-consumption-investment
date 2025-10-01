import numpy as np


def summarize_terminal_wealth(terminal_wealth: np.ndarray) -> dict[str, float]:
    """Basic stats for W_T."""
    return {
        "mean": float(np.mean(terminal_wealth)),
        "median": float(np.median(terminal_wealth)),
        "p05": float(np.quantile(terminal_wealth, 0.05)),
        "p95": float(np.quantile(terminal_wealth, 0.95)),
        "prob_ruin": float(np.mean(terminal_wealth <= 0.0)),
    }


def consumption_sharpe(consumption_rate: np.ndarray) -> float:
    """
    Annualized Sharpe of the *consumption rate* time series averaged across paths.
    consumption_rate shape: (n_steps, n_paths)
    """
    mean_series = np.mean(consumption_rate, axis=1)
    mu = float(np.mean(mean_series))
    sigma = float(np.std(mean_series, ddof=1))
    if sigma == 0.0:
        return 0.0
    return mu / sigma
