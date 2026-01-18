from dataclasses import dataclass

@dataclass(frozen=True)
class ModelParams:
    # Market Parameters
    r: float = 0.02       # risk-free rate (annual)
    mu: float = 0.06      # risky drift (annual)
    sigma: float = 0.20   # risky volatility (annual)
    
    # Agent Parameters
    rho: float = 0.03     # discount rate (annual)
    gamma: float = 2.0    # CRRA risk aversion
    
    # Horizon (REQUIRED for main.py)
    T: float = 30.0       # years
    
    # Constraints & Terminal
    cbar: float = 0.08    # max consumption rate (optional)
    kappa: float = 0.0    # terminal-wealth weight

@dataclass(frozen=True)
class SimConfig:
    # Simulation settings
    dt: float = 1 / 252   # years per step (daily)
    n_paths: int = 20_000
    seed: int | None = 7
    absorb_at_zero: bool = True