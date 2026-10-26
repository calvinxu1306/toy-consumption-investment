import math
from dataclasses import dataclass, fields


def _require(ok: bool, message: str):
    if not ok:
        raise ValueError(message)


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
    pi_max: float = 1.5   # max risky weight (leverage limit), applied to every policy
    kappa: float = 0.0    # terminal-wealth weight

    def __post_init__(self):
        # Reject values the model is undefined for, instead of crashing deep in
        # the solver or quietly printing nan / misleading results
        for f in fields(self):
            value = getattr(self, f.name)
            _require(math.isfinite(value), f"{f.name} must be a finite number (got {value})")
        _require(self.gamma > 0, f"gamma must be > 0 (got {self.gamma})")
        _require(self.sigma >= 0, f"sigma must be >= 0 (got {self.sigma})")
        _require(self.T > 0, f"T must be > 0 (got {self.T})")
        _require(self.pi_max >= 0, f"pi_max must be >= 0 (got {self.pi_max})")
        _require(self.kappa >= 0, f"kappa must be >= 0 (got {self.kappa})")

@dataclass(frozen=True)
class SimConfig:
    # Simulation settings
    dt: float = 1 / 12    # years per step (monthly, same as the DP solver's default)
    n_paths: int = 20_000
    seed: int | None = 7
    absorb_at_zero: bool = True

    def __post_init__(self):
        _require(math.isfinite(self.dt) and self.dt > 0, f"dt must be > 0 (got {self.dt})")
        # Two paths are the minimum for a Monte Carlo standard error
        _require(self.n_paths >= 2, f"n_paths must be >= 2 (got {self.n_paths})")
        _require(self.seed is None or self.seed >= 0, f"seed must be >= 0 (got {self.seed})")