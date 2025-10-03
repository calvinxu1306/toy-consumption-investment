import os
import csv
import numpy as np

from src.models.params import ModelParams, SimConfig
from src.models.policies import constant_policy
from src.models.simulate_wealth import simulate


def make_policy(pi_val: float, alpha_val: float):
    """Factory for constant (pi, alpha) policies."""
    def policy(wealth, t, params):
        return constant_policy(
            wealth, t, params, pi_const=pi_val, alpha_consume=alpha_val
        )
    return policy


def main():
    os.makedirs("data", exist_ok=True)

    params = ModelParams()
    sim = SimConfig(n_paths=12000, seed=4)

    pi_vals = np.linspace(0.3, 0.7, 5)    # 0.30, 0.40, 0.50, 0.60, 0.70
    alpha_vals = np.linspace(0.02, 0.06, 9)

    out_path = "data/policy_grid_local.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "pi", "alpha", "DU",
            "mean_W_T", "median_W_T", "p05_W_T", "p95_W_T", "prob_ruin_T"
        ])
        for pi in pi_vals:
            for a in alpha_vals:
                policy = make_policy(pi, a)
                res = simulate(params, sim, policy)
                W_T = res["W"][-1]
                w.writerow([
                    float(pi), float(a), float(res["DU"]),
                    float(W_T.mean()), float(np.median(W_T)),
                    float(np.quantile(W_T, 0.05)),
                    float(np.quantile(W_T, 0.95)),
                    float((W_T <= 0.0).mean()),
                ])
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
