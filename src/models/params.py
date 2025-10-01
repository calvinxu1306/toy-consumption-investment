from dataclasses import dataclass


@dataclass(frozen=True)
class ModelParams:
    r: float = 0.02
    mu: float = 0.06
    sigma: float = 0.20
    rho: float = 0.03
    gamma: float = 2.0
    cbar: float = 0.08
    kappa: float = 0.0


@dataclass(frozen=True)
class SimConfig:
    T: float = 30.0
    dt: float = 1/252
    n_paths: int = 20000
    seed: int | None = 7
    absorb_at_zero: bool = True
