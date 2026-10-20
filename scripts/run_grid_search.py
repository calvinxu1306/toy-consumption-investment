import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import numpy as np
import matplotlib.pyplot as plt

from src.models.dynamics import simulate, WEALTH_FLOOR
from src.models.metrics import summarize_terminal_wealth
from src.models.params import ModelParams, SimConfig
from src.models.policies import constant_policy, get_merton_values

def make_policy(pi_val: float, alpha_val: float):
    """Factory for constant (pi, alpha) policies."""
    def policy(wealth, t, params):
        return constant_policy(
            wealth, t, params, pi_const=pi_val, alpha_consume=alpha_val
        )
    return policy

def run_single_demo(params, sim):
    """
    Runs a single simulation to generate plots (migrated from old simulate_wealth.py).
    """
    print("\n--- Running Single Demo Simulation ---")
    # Baseline policy: 50% risky, 4% consumption
    demo_policy = make_policy(pi_val=0.5, alpha_val=0.04)
    out = simulate(params, sim, demo_policy)
    
    t = out["times"]
    w = out["W"]  # (n_paths, n_steps + 1)

    # 1. Wealth Quantiles
    q05 = np.quantile(w, 0.05, axis=0)
    q50 = np.quantile(w, 0.50, axis=0)
    q95 = np.quantile(w, 0.95, axis=0)

    plt.figure()
    plt.plot(t, q50, label="median")
    plt.fill_between(t, q05, q95, alpha=0.25, label="5-95% band")
    plt.xlabel("Years")
    plt.ylabel("Wealth")
    plt.title("Wealth Quantiles (Demo)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/wealth_quantiles.png", dpi=140)
    print("Saved: data/wealth_quantiles.png")

    # 2. Ruin Curve
    ruin_curve = (w <= WEALTH_FLOOR).mean(axis=0)
    plt.figure()
    plt.plot(t, ruin_curve)
    plt.xlabel("Years")
    plt.ylabel("Ruin probability")
    plt.title("Ruin Probability Over Time")
    plt.tight_layout()
    plt.savefig("data/ruin_curve.png", dpi=140)
    print("Saved: data/ruin_curve.png")

    # 3. Sample Paths
    plt.figure()
    sel = min(50, w.shape[0])
    plt.plot(t, w[:sel].T)
    plt.title("Wealth Sample Paths")
    plt.xlabel("Years")
    plt.ylabel("Wealth")
    plt.tight_layout()
    plt.savefig("data/wealth_paths.png", dpi=140)
    print("Saved: data/wealth_paths.png")

def main():
    os.makedirs("data", exist_ok=True)
    
    params = ModelParams()
    # Use fewer paths/seed for grid search speed, or more for accuracy
    sim = SimConfig(n_paths=5000, seed=4) 

    # --- Part 1: Grid Search ---
    print(f"Starting Grid Search (N={sim.n_paths})...")
    pi_vals = np.linspace(0.3, 0.7, 5)    
    alpha_vals = np.linspace(0.02, 0.10, 17)  # finite T pushes the best alpha above alpha*

    out_path = "data/policy_grid_local.csv"
    
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "pi", "alpha", "DU",
            "mean_W_T", "median_W_T", "p05_W_T", "p95_W_T", "prob_ruin_T"
        ])
        
        total_runs = len(pi_vals) * len(alpha_vals)
        count = 0
        best = None
        
        for pi in pi_vals:
            for a in alpha_vals:
                count += 1
                policy = make_policy(pi, a)
                res = simulate(params, sim, policy)
                
                stats = summarize_terminal_wealth(res["W"][:, -1])
                w.writerow([
                    float(pi), float(a), float(res["DU"]),
                    stats["mean"], stats["median"],
                    stats["p05"], stats["p95"], stats["prob_ruin"],
                ])
                if best is None or res["DU"] > best[2]:
                    best = (pi, a, res["DU"])
                if count % 5 == 0:
                    print(f"  Processed {count}/{total_runs}...")
                    
    print(f"Grid Search Complete. Results: {out_path}")

    # Compare the best constant policy with the infinite-horizon Merton values.
    # Over a finite horizon the best constant alpha sits above alpha*, because
    # there is no reason to preserve wealth past T.
    pi_star, alpha_star = get_merton_values(params)
    print(f"  Best on grid:          pi={best[0]:.2f}, alpha={best[1]:.3f} (DU={best[2]:.2f})")
    print(f"  Merton (infinite T):   pi={pi_star:.2f}, alpha={alpha_star:.3f}")

    # --- Part 2: Generate Plots for one baseline ---
    run_single_demo(params, sim)

if __name__ == "__main__":
    main()