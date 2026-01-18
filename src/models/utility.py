import numpy as np

_EPS = 1e-2 

def crra(c: np.ndarray | float, gamma: float):
    c_arr = np.maximum(np.asarray(c, dtype=float), _EPS)
    if abs(gamma - 1.0) < 1e-12:
        return np.log(c_arr)
    return (c_arr ** (1 - gamma)) / (1 - gamma)


def terminal_crra(w: np.ndarray | float, gamma: float, kappa: float):
    if kappa == 0.0:
        return np.zeros_like(np.asarray(w, dtype=float))
    w_arr = np.maximum(np.asarray(w, dtype=float), _EPS)
    if abs(gamma - 1.0) < 1e-12:
        return kappa * np.log(w_arr)
    return kappa * (w_arr ** (1 - gamma)) / (1 - gamma)