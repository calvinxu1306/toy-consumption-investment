import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

from src.models.params import ModelParams
from src.models.policies import finite_horizon_merton_policy
from src.solvers.dp_solver import DPSolverFiniteHorizon

def plot_heatmaps(solver, save_dir="data"):
    """
    Visualizes the optimal policies (Pi and Alpha) as heatmaps (Time vs Wealth).
    With CRRA utility the optimum does not depend on wealth, so the bands
    should be vertical; anything else is a numerical artifact.
    """
    os.makedirs(save_dir, exist_ok=True)
    t = solver.time_grid[:-1]

    # Consumption rates span 0.01 to ~1/dt near T, so use a log color scale
    alpha_map = np.maximum(solver.opt_c, 1e-3)
    panels = [
        ("Optimal Risky Weight (Pi) over Time", solver.opt_pi, "viridis",
         Normalize(0.0, solver.params.pi_max), "dp_policy_pi.png"),
        ("Optimal Consumption Rate (Alpha) over Time", alpha_map, "magma",
         LogNorm(alpha_map.min(), alpha_map.max()), "dp_policy_alpha.png"),
    ]
    for title, table, cmap, norm, filename in panels:
        fig, ax = plt.subplots(figsize=(10, 6))
        mesh = ax.pcolormesh(t, solver.w_grid, table.T, cmap=cmap, norm=norm, shading="auto")
        fig.colorbar(mesh, ax=ax)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Years")
        ax.set_ylabel("Wealth")
        fig.savefig(f"{save_dir}/{filename}")
        plt.close(fig)
        print(f"Saved: {save_dir}/{filename}")

def plot_vs_closed_form(solver, save_dir="data"):
    """
    Compares the DP policy at W = 1 with the closed-form finite-horizon optimum.
    The DP's pi dips in the final months: in discrete time, consumption leaves
    the account during the step while risk is taken on starting wealth, so the
    optimal weight is about pi* * (1 - alpha * dt), which falls to 0 when alpha = 1/dt.
    """
    os.makedirs(save_dir, exist_ok=True)
    t = solver.time_grid[:-1]
    i = np.argmin(np.abs(solver.w_grid - 1.0))
    w = np.array([1.0])
    exact = np.array([finite_horizon_merton_policy(w, tk, solver.params) for tk in t])[:, :, 0]

    fig, (ax_pi, ax_c) = plt.subplots(1, 2, figsize=(13, 5))
    ax_pi.plot(t, solver.opt_pi[:, i], "r-", label="DP")
    ax_pi.plot(t, exact[:, 0], "k:", label="Closed form")
    ax_pi.set_title("Risky Weight (Pi)")
    ax_c.semilogy(t, solver.opt_c[:, i], "r-", label="DP")
    ax_c.semilogy(t, exact[:, 1], "k:", label="Closed form")
    ax_c.set_title("Consumption Rate (Alpha), log scale")
    for ax in (ax_pi, ax_c):
        ax.set_xlabel("Years")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(f"{save_dir}/dp_vs_closed_form.png")
    plt.close(fig)
    print(f"Saved: {save_dir}/dp_vs_closed_form.png")

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("--- Initializing DP Solver ---")
    params = ModelParams()

    # Monthly steps over the horizon (the solver's default)
    solver = DPSolverFiniteHorizon(params)
    solver.train_backward()

    print("\n--- Generating Visualization ---")
    plot_heatmaps(solver)
    plot_vs_closed_form(solver)

    print("\nDP Run Complete. Check 'data/' for heatmaps.")

if __name__ == "__main__":
    main()
