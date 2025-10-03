import numpy as np
import matplotlib.pyplot as plt
import os

from src.models.params import ModelParams, SimConfig
from src.models.policies import constant_policy
from src.models.dynamics import step_wealth
from src.models.utility import crra, terminal_crra

RUIN_EPS = 1e-6


def simulate(
        params: ModelParams = ModelParams(),
        sim: SimConfig = SimConfig(),
        policy=lambda w_k, t_k, p: constant_policy(
            w_k, t_k, p, pi_const=0.5, alpha_consume=0.04
        ),
        w0: float = 1.0,
):
    n_steps = int(round(sim.T / sim.dt))
    time_grid = np.linspace(0.0, sim.T, n_steps + 1)

    rng = np.random.default_rng(sim.seed)
    z = rng.standard_normal((n_steps, sim.n_paths))

    wealth = np.zeros((n_steps + 1, sim.n_paths))
    wealth[0] = w0
    cons = np.zeros((n_steps, sim.n_paths))
    risky = np.zeros((n_steps, sim.n_paths))

    disc = np.exp(-params.rho * time_grid[:-1])

    du = np.zeros(sim.n_paths)
    alive = np.ones(sim.n_paths, dtype=bool)  # all start alive

    for k in range(n_steps):
        # controls for all paths (policy can zero out c when wealth==0)
        pi_k, c_k = policy(wealth[k], time_grid[k], params)
        risky[k] = pi_k
        cons[k] = c_k

        # accumulate utility ONLY for alive paths
        if np.any(alive):
            du[alive] += disc[k] * crra(c_k[alive], params.gamma) * sim.dt

        # step wealth ONLY for alive paths (dead ones stay at ~0)
        next_w = wealth[k + 1]     # alias for clarity
        next_w[:] = wealth[k]      # start from current
        if np.any(alive):
            next_w[alive] = step_wealth(
                wealth[k][alive],
                pi_k[alive],
                c_k[alive],
                params,
                sim.dt,
                z[k][alive],
                absorb_at_zero=sim.absorb_at_zero,
            )

        # update alive mask AFTER stepping (ruin if near-zero)
        alive = next_w > RUIN_EPS

        du += np.exp(-params.rho * time_grid[-1]) * terminal_crra(
            wealth[-1], params.gamma, params.kappa
        )

    return {
        "times": time_grid,
        "W": wealth,
        "C": cons,
        "PI": risky,
        "DU_vec": du,
        "DU": float(np.mean(du)),
    }


if __name__ == "__main__":
    out = simulate()
    t = out["times"]
    w = out["W"]

    os.makedirs("data", exist_ok=True)

    # Wealth quantiles
    q05 = np.quantile(w, 0.05, axis=1)
    q50 = np.quantile(w, 0.50, axis=1)
    q95 = np.quantile(w, 0.95, axis=1)

    plt.figure()
    plt.plot(t, q50, label="median")
    plt.fill_between(t, q05, q95, alpha=0.25, label="5-95% band")
    plt.xlabel("Years")
    plt.ylabel("Wealth")
    plt.title("Wealth Quantiles")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/wealth_quantiles.png", dpi=140)

    # Ruin curve (fraction of paths with W<=0 at end of simulation)
    ruin_curve = (w <= RUIN_EPS).mean(axis=1)

    plt.figure()
    plt.plot(t, ruin_curve)
    plt.xlabel("Years")
    plt.ylabel("Ruin probability")
    plt.title("Ruin Probability Over Time")
    plt.tight_layout()
    plt.savefig("data/ruin_curve.png", dpi=140)

    # Sample paths
    plt.figure()
    sel = min(50, w.shape[1])
    plt.plot(t, w[:, :sel])
    plt.title("Wealth Sample Paths (baseline policy)")
    plt.xlabel("Years")
    plt.ylabel("Wealth")
    plt.tight_layout()
    plt.savefig("data/wealth_paths.png", dpi=140)
    plt.show()

    # Scoreboard
    W_T = w[-1]
    scoreboard = {
        "DU": float(out["DU"]),
        "mean_W_T": float(W_T.mean()),
        "median_W_T": float(np.median(W_T)),
        "p05_W_T": float(np.quantile(W_T, 0.05)),
        "p95_W_T": float(np.quantile(W_T, 0.95)),
        "prob_ruin_T": float((W_T <= 0.0).mean()),
    }
    print("Run summary:", scoreboard)
