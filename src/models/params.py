from dataclasses import dataclass


@dataclass(frozen=True)
class ModelParams:
    r: float = 0.02       # risk-free rate (annual)
    mu: float = 0.06      # risky drift (annual)
    sigma: float = 0.20   # risky volatility (annual)
    rho: float = 0.03     # discount rate (annual)
    gamma: float = 2.0    # CRRA risk aversion
    cbar: float = 0.08    # max consumption rate as fraction of W per year
    kappa: float = 0.0    # terminal-wealth weight (0 disables)


@dataclass(frozen=True)
class SimConfig:
    T: float = 30.0       # years
    dt: float = 1 / 252   # years per step (daily)
    n_paths: int = 20_000
    seed: int | None = 7
    absorb_at_zero: bool = True
