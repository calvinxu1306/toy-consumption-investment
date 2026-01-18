import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

from src.models.params import ModelParams, SimConfig
from src.models.dynamics import simulate
from src.models.policies import constant_policy
from src.solvers.dp_solver import DPSolverFiniteHorizon

def parse_arguments():
    """
    Defines the available command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Optimal Consumption & Investment Simulator (CLI)")
    
    # Financial Parameters
    parser.add_argument("--mu", type=float, default=0.06, help="Expected stock return (Drift)")
    parser.add_argument("--r", type=float, default=0.02, help="Risk-free rate")
    parser.add_argument("--sigma", type=float, default=0.20, help="Volatility (Standard Deviation)")
    parser.add_argument("--gamma", type=float, default=2.0, help="Risk Aversion (Higher = More conservative)")
    parser.add_argument("--T", type=float, default=30.0, help="Investment Horizon (Years)")
    parser.add_argument("--rho", type=float, default=0.03, help="Discount Rate")
    
    # Simulation Settings
    parser.add_argument("--paths", type=int, default=5000, help="Number of simulation paths")
    parser.add_argument("--output", type=str, default="data", help="Output directory for plots")

    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # 1. Update Parameters from Arguments
    print(f"--- Configuration ---")
    print(f"Risk Aversion (Gamma): {args.gamma}")
    print(f"Volatility (Sigma):    {args.sigma}")
    print(f"Horizon (T):           {args.T} years")
    
    params = ModelParams(
        mu=args.mu, r=args.r, sigma=args.sigma, 
        gamma=args.gamma, rho=args.rho, T=args.T
    )
    
    # 2. Train the Smart Agent
    print("\n--- 1. Training Smart Agent (DP Solver) ---")
    # We adjust grid size dynamically based on horizon to keep it reasonably fast
    n_steps = int(args.T * 10) # 10 steps per year
    solver = DPSolverFiniteHorizon(params, w_min=0.01, w_max=15.0, 
                                   n_w_points=100, n_time_steps=n_steps)
    solver.initialize_terminal_condition()
    solver.train_backward()
    
    smart_policy_func = solver.get_policy_function()
    
    # 3. Define Policies
    def smart_wrapper(w, t, p):
        return smart_policy_func(w, t, p)

    # Calculate Merton Benchmark for comparison
    merton_pi = (args.mu - args.r) / (args.gamma * args.sigma**2)
    print(f"   (Merton Benchmark Pi*: {merton_pi:.2f})")
    
    def baseline_wrapper(w, t, p):
        # Use simple Merton constant for baseline
        return constant_policy(w, t, p, pi_const=merton_pi, alpha_consume=0.04)

    # 4. Run Simulation
    print("\n--- 2. Running Simulations ---")
    sim_cfg = SimConfig(n_paths=args.paths, seed=42)
    
    res_base = simulate(params, sim_cfg, baseline_wrapper)
    res_smart = simulate(params, sim_cfg, smart_wrapper)
    
    # 5. Compare Results
    du_base = res_base["DU"]
    du_smart = res_smart["DU"]
    imp = ((du_smart - du_base) / abs(du_base)) * 100
    
    print("\n" + "="*30)
    print(f"FINAL SCORE (Expected Utility)")
    print("="*30)
    print(f"Baseline:    {du_base:.4f}")
    print(f"Smart Agent: {du_smart:.4f}")
    print(f"Improvement: {imp:+.2f}%")
    print("="*30)

    # 6. Save Plots
    os.makedirs(args.output, exist_ok=True)
    
    # Plot Wealth
    plt.figure(figsize=(10, 6))
    
    # Check shapes for debugging (optional, but helpful)
    # print(f"DEBUG: Times shape: {res_base['times'].shape}")
    # print(f"DEBUG: Wealth shape: {res_base['W'].shape}")

    # Calculate mean wealth across paths (axis 0 is paths, axis 1 is time steps)
    avg_w_base = res_base["W"].mean(axis=0)  
    avg_w_smart = res_smart["W"].mean(axis=0)
    
    plt.plot(res_base["times"], avg_w_base, 'b--', label=f"Baseline (Merton pi={merton_pi:.2f})")
    plt.plot(res_smart["times"], avg_w_smart, 'r-', label="Smart Agent (DP)")
    
    plt.title(f"Wealth Trajectory (Gamma={args.gamma})")
    plt.xlabel("Years")
    plt.ylabel("Average Wealth")
    plt.legend()
    plt.grid(alpha=0.3)
    
    filename = f"{args.output}/cli_wealth_g{args.gamma}_s{args.sigma}.png"
    plt.savefig(filename)
    print(f"\nResults saved to: {filename}")

if __name__ == "__main__":
    main()