import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.models.params import ModelParams, SimConfig
from src.models.dynamics import simulate
from src.models.policies import constant_policy
from src.solvers.dp_solver import DPSolverFiniteHorizon

def main():
    # 1. Train the "Smart Agent" (DP Solver)
    print("--- Training DP Solver (The Smart Agent) ---")
    params = ModelParams()
    # Finer grid for better competition results
    solver = DPSolverFiniteHorizon(params, w_min=0.01, w_max=15.0, 
                                   n_w_points=200, n_time_steps=300)
    solver.initialize_terminal_condition()
    solver.train_backward()
    
    # Get the interpolation function
    smart_policy_func = solver.get_policy_function()
    
    # Wrap it to match the simulate() signature: policy(w, t, params)
    def smart_policy_wrapper(w, t, p):
        return smart_policy_func(w, t, p)

    # 2. Define the "Baseline Agent" (Merton Constant)
    # Based on your Grid Search, pi=0.5 and alpha=0.04 was roughly best
    def baseline_policy_wrapper(w, t, p):
        return constant_policy(w, t, p, pi_const=0.5, alpha_consume=0.04)

    # 3. Run Head-to-Head Simulation
    print("\n--- Running Head-to-Head Simulation ---")
    sim_cfg = SimConfig(n_paths=10000, seed=42) # Fair comparison: same seed
    
    print("Simulating Baseline Agent...")
    res_base = simulate(params, sim_cfg, baseline_policy_wrapper)
    
    print("Simulating Smart Agent...")
    res_smart = simulate(params, sim_cfg, smart_policy_wrapper)
    
    # 4. The Scoreboard
    du_base = res_base["DU"]
    du_smart = res_smart["DU"]
    improvement = ((du_smart - du_base) / abs(du_base)) * 100
    
    print("\n" + "="*40)
    print("       FINAL SCOREBOARD (Expected Utility)")
    print("="*40)
    print(f"Baseline (Constant):  {du_base:.4f}")
    print(f"Smart Agent (DP):     {du_smart:.4f}")
    print(f"Improvement:          {improvement:+.2f}%")
    print("="*40)
    
    # 5. Visualization: Average Wealth Paths
    t_grid = res_base["times"]
    avg_w_base = res_base["W"].mean(axis=1)
    avg_w_smart = res_smart["W"].mean(axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_grid, avg_w_base, 'b--', label="Baseline (Constant)", alpha=0.7)
    plt.plot(t_grid, avg_w_smart, 'r-', label="Smart Agent (DP)", linewidth=2)
    
    plt.title("Comparison: Average Wealth Trajectory")
    plt.xlabel("Years")
    plt.ylabel("Average Wealth")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs("data", exist_ok=True)
    plt.savefig("data/comparison_wealth.png")
    print("\nSaved plot to data/comparison_wealth.png")
    
    # Visualization: Average Consumption Paths
    avg_c_base = res_base["C"].mean(axis=1)
    avg_c_smart = res_smart["C"].mean(axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_grid[:-1], avg_c_base, 'b--', label="Baseline Consumption", alpha=0.7)
    plt.plot(t_grid[:-1], avg_c_smart, 'r-', label="Smart Consumption", linewidth=2)
    
    plt.title("Comparison: Average Consumption Trajectory")
    plt.xlabel("Years")
    plt.ylabel("Consumption Rate ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("data/comparison_consumption.png")
    print("Saved plot to data/comparison_consumption.png")

if __name__ == "__main__":
    main()