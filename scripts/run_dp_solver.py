import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.params import ModelParams
from src.solvers.dp_solver import DPSolverFiniteHorizon

def plot_heatmaps(solver, save_dir="data"):
    """
    Visualizes the optimal policies (Pi and Alpha) as heatmaps (Time vs Wealth).
    """
    # 1. Prepare Data for Heatmap
    # V, opt_pi, opt_c are shape (n_t, n_w) or similar.
    # We transpose them so Time is on X-axis, Wealth on Y-axis
    pi_map = solver.opt_pi.T  # Shape: (n_w, n_t)
    c_map = solver.opt_c.T    # Shape: (n_w, n_t)
    
    # Create extensive labels for axes
    y_ticks = np.round(solver.w_grid, 2)
    x_ticks = np.round(solver.time_grid[:-1], 1)
    
    # reduce tick density for cleanliness
    y_step = len(y_ticks) // 10
    x_step = len(x_ticks) // 10

    # --- Plot Risky Weight (Pi) ---
    plt.figure(figsize=(10, 6))
    sns.heatmap(pi_map, cmap="viridis", vmin=0.0, vmax=1.0,
                xticklabels=x_step, yticklabels=y_step)
    
    plt.title("Optimal Risky Weight (Pi) over Time")
    plt.xlabel("Time Steps (Indices)")
    plt.ylabel("Wealth Grid (Indices)")
    plt.gca().invert_yaxis() # Put high wealth at top
    plt.savefig(f"{save_dir}/dp_policy_pi.png")
    print(f"Saved: {save_dir}/dp_policy_pi.png")

    # --- Plot Consumption Rate (Alpha) ---
    plt.figure(figsize=(10, 6))
    sns.heatmap(c_map, cmap="magma", vmin=0.0, vmax=0.20,
                xticklabels=x_step, yticklabels=y_step)
    
    plt.title("Optimal Consumption Rate (Alpha) over Time")
    plt.xlabel("Time Steps (Indices)")
    plt.ylabel("Wealth Grid (Indices)")
    plt.gca().invert_yaxis()
    plt.savefig(f"{save_dir}/dp_policy_alpha.png")
    print(f"Saved: {save_dir}/dp_policy_alpha.png")

def main():
    print("--- Initializing DP Solver ---")
    params = ModelParams()
    
    # Use a coarser grid for testing (faster), finer for final results
    # n_time_steps=300 means roughly monthly updates over 30 years
    solver = DPSolverFiniteHorizon(params, n_w_points=50, n_time_steps=300)
    
    solver.initialize_terminal_condition()
    solver.train_backward()
    
    print("\n--- Generating Visualization ---")
    plot_heatmaps(solver)
    
    print("\nDP Run Complete. Check 'data/' for heatmaps.")

if __name__ == "__main__":
    main()