"""
Train -> simulate -> score pipeline shared by main.py, app.py and the scripts.

Three policies are compared on the same random shocks:
  - Static Merton baseline: infinite-horizon pi* and alpha*, ignoring the horizon
  - DP agent: the backward-induction solution
  - Closed-form optimum: the exact finite-horizon solution (the ceiling)
"""
from dataclasses import replace

import matplotlib.pyplot as plt

from src.models.dynamics import simulate
from src.models.metrics import certainty_equivalent
from src.models.params import ModelParams, SimConfig
from src.models.policies import finite_horizon_merton_policy, merton_policy
from src.solvers.dp_solver import DPSolverFiniteHorizon

BASELINE = "Static Merton"
DP_AGENT = "DP Agent"
OPTIMUM = "Closed-form Optimum"

STYLES = {
    BASELINE: dict(color="tab:blue", linestyle="--"),
    DP_AGENT: dict(color="tab:red", linewidth=2),
    OPTIMUM: dict(color="black", linestyle=":"),
}


def train_dp_agent(params: ModelParams, **grid_kwargs) -> DPSolverFiniteHorizon:
    """Solves the DP on the default (monthly) grid unless overridden."""
    solver = DPSolverFiniteHorizon(params, **grid_kwargs)
    solver.train_backward()
    return solver


def compare_policies(params: ModelParams, sim: SimConfig,
                     solver: DPSolverFiniteHorizon) -> dict[str, dict]:
    """
    Simulates all three policies with the same seed. Each result is simulate()'s
    output plus "CE", the certainty-equivalent consumption rate.
    """
    # Simulate on the grid the DP was solved on, so its policy is scored in the
    # same discrete-time model it optimized
    sim = replace(sim, dt=solver.dt)
    policies = {
        BASELINE: merton_policy,
        DP_AGENT: solver.get_policy_function(),
        OPTIMUM: finite_horizon_merton_policy,
    }
    results = {}
    for name, policy in policies.items():
        res = simulate(params, sim, policy)
        res["CE"] = certainty_equivalent(res["DU"], params, res["times"])
        results[name] = res
    return results


def gain_vs_baseline(results: dict[str, dict], name: str) -> float:
    """Consumption-equivalent gain of `name` over the baseline (0.1 = +10%)."""
    base_ce = results[BASELINE]["CE"]
    return results[name]["CE"] / base_ce - 1 if base_ce > 0 else float("inf")


def dp_gap(results: dict[str, dict]) -> float:
    """Relative distance between the DP agent and the closed-form optimum (CE units)."""
    return abs(1 - results[DP_AGENT]["CE"] / results[OPTIMUM]["CE"])


def format_scoreboard(results: dict[str, dict]) -> str:
    lines = [f"{'Policy':<21}{'Expected utility':>24}{'CE consumption':>16}{'vs baseline':>13}"]
    for name, res in results.items():
        du = f"{res['DU']:.4g} +/- {res['DU_se']:.2g}"
        lines.append(f"{name:<21}{du:>24}{res['CE']:>16.4f}"
                     f"{gain_vs_baseline(results, name):>+13.1%}")
    lines.append(f"\nDP agent is within {dp_gap(results):.2%} of the closed-form optimum "
                 f"(certainty-equivalent consumption).")
    return "\n".join(lines)


def plot_comparison(results: dict[str, dict], title: str = ""):
    """Average wealth and consumption over time for each policy."""
    fig, (ax_w, ax_c) = plt.subplots(1, 2, figsize=(13, 5))
    for name, res in results.items():
        t = res["times"]
        ax_w.plot(t, res["W"].mean(axis=0), label=name, **STYLES.get(name, {}))
        ax_c.plot(t[:-1], res["C"].mean(axis=0), label=name, **STYLES.get(name, {}))

    ax_w.set_title("Average Wealth")
    ax_c.set_title("Average Consumption Rate")
    for ax in (ax_w, ax_c):
        ax.set_xlabel("Years")
        ax.grid(alpha=0.3)
        ax.legend()
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
