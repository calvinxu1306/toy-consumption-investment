import argparse
import logging
import os

import matplotlib.pyplot as plt

from src.experiment import compare_policies, format_scoreboard, plot_comparison, train_dp_agent
from src.models.params import ModelParams, SimConfig
from src.models.policies import get_merton_values

def parse_arguments():
    """
    Defines the available command-line arguments.
    """
    defaults = ModelParams()
    parser = argparse.ArgumentParser(description="Optimal Consumption & Investment Simulator (CLI)")

    # Financial Parameters
    parser.add_argument("--mu", type=float, default=defaults.mu, help="Expected stock return (Drift)")
    parser.add_argument("--r", type=float, default=defaults.r, help="Risk-free rate")
    parser.add_argument("--sigma", type=float, default=defaults.sigma, help="Volatility (Standard Deviation)")
    parser.add_argument("--gamma", type=float, default=defaults.gamma, help="Risk Aversion (Higher = More conservative)")
    parser.add_argument("--T", type=float, default=defaults.T, help="Investment Horizon (Years)")
    parser.add_argument("--rho", type=float, default=defaults.rho, help="Discount Rate")
    parser.add_argument("--kappa", type=float, default=defaults.kappa, help="Bequest weight on terminal wealth (0 = none)")

    # Simulation Settings
    parser.add_argument("--paths", type=int, default=5000, help="Number of simulation paths")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (all policies see the same shocks)")
    parser.add_argument("--output", type=str, default="data", help="Output directory for plots")

    return parser.parse_args()

def main():
    args = parse_arguments()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # 1. Update Parameters from Arguments
    params = ModelParams(
        mu=args.mu, r=args.r, sigma=args.sigma,
        gamma=args.gamma, rho=args.rho, T=args.T, kappa=args.kappa
    )
    pi_star, alpha_star = get_merton_values(params)

    print("--- Configuration ---")
    print(f"Market:  mu={params.mu}, r={params.r}, sigma={params.sigma}")
    print(f"Agent:   gamma={params.gamma}, rho={params.rho}, kappa={params.kappa}")
    print(f"Horizon: T={params.T} years")
    print(f"Merton:  pi*={pi_star:.3f}, alpha*={alpha_star:.4f}")

    # 2. Train the Smart Agent
    print("\n--- 1. Training Smart Agent (DP Solver) ---")
    solver = train_dp_agent(params)

    # 3. Run Simulation
    print("\n--- 2. Running Simulations ---")
    results = compare_policies(params, SimConfig(n_paths=args.paths, seed=args.seed), solver)

    # 4. Compare Results
    print("\n" + "=" * 74)
    print(format_scoreboard(results))
    print("=" * 74)

    # 5. Save Plots
    os.makedirs(args.output, exist_ok=True)
    tag = (f"g{params.gamma}_s{params.sigma}_mu{params.mu}_r{params.r}"
           f"_rho{params.rho}_T{params.T}_k{params.kappa}")
    fig = plot_comparison(results, title=f"gamma={params.gamma}, sigma={params.sigma}, T={params.T}")
    filename = os.path.join(args.output, f"cli_wealth_{tag}.png")
    fig.savefig(filename)
    plt.close(fig)
    print(f"\nResults saved to: {filename}")

if __name__ == "__main__":
    main()
