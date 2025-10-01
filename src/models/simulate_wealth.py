import numpy as np
import matplotlib.pyplot as plt

from src.models.params import ModelParams, SimConfig
from src.models.policies import constant_policy
from src.models.dynamics import step_wealth
from src.models.utility import crra, terminal_crra


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

    for k in range(n_steps):
        pi_k, c_k = policy(wealth[k], time_grid[k], params)
        risky[k] = pi_k
        cons[k] = c_k

        du += disc[k] * crra(c_k, params.gamma) * sim.dt
        wealth[k + 1] = step_wealth(
            wealth[k],
            pi_k,
            c_k,
            params,
            sim.dt,
            z[k],
            absorb_at_zero=sim.absorb_at_zero,
        )

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

    # plot a handful of sample wealth paths
    plt.figure()
    sel = min(50, w.shape[1])
    plt.plot(t, w[:, :sel])
    plt.title("Wealth Sample Paths (baseline policy)")
    plt.xlabel("Years")
    plt.ylabel("Wealth")
    plt.tight_layout()
    plt.savefig("data/wealth_paths.png", dpi=140)
    plt.show()

    print("E[discounted utility]:", out["DU"])
