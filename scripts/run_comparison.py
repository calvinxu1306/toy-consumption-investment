import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

import matplotlib.pyplot as plt
from src.experiment import compare_policies, format_scoreboard, plot_comparison, train_dp_agent
from src.models.params import ModelParams, SimConfig

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # 1. Train the "Smart Agent" (DP Solver)
    print("--- Training DP Solver (The Smart Agent) ---")
    params = ModelParams()
    solver = train_dp_agent(params)

    # 2. Run Head-to-Head Simulation against the static Merton baseline and
    # the closed-form optimum (same seed: every policy sees the same shocks)
    print("\n--- Running Head-to-Head Simulation ---")
    results = compare_policies(params, SimConfig(n_paths=10000, seed=42), solver)

    # 3. The Scoreboard
    print("\n" + "=" * 74)
    print("FINAL SCOREBOARD")
    print("=" * 74)
    print(format_scoreboard(results))
    print("=" * 74)

    # 4. Visualization: Average Wealth and Consumption Paths
    os.makedirs("data", exist_ok=True)
    fig = plot_comparison(results, title="Comparison: Static Merton vs DP Agent vs Closed-form Optimum")
    fig.savefig("data/comparison.png")
    plt.close(fig)
    print("\nSaved plot to data/comparison.png")

if __name__ == "__main__":
    main()
